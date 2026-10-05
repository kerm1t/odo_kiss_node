// ply_odom - KISS-ICP odometry over a sequence of PLY scans.
//
//   ply_odom <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m] [-v voxel_m]
//
// A directory is scanned for *.ply and sorted naturally (frame_2 < frame_10);
// files given explicitly keep their order. Writes one pose per frame in KITTI
// format (3x4 row-major, 12 values per line). Poses are T_0_i with
// p_0 = T_0_i * p_i, i.e. the sensor pose at frame i in the first frame.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <happly.h>
#include <kiss_icp/pipeline/KissICP.hpp>

namespace fs = std::filesystem;
using Cloud = std::vector<Eigen::Vector3d>;
using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Cloud loadPly(const std::string &path) {
    happly::PLYData ply(path);
    const auto v = ply.getVertexPositions();  // float or double x/y/z
    Cloud pts;
    pts.reserve(v.size());
    for (const auto &p : v) {
        if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
            pts.emplace_back(p[0], p[1], p[2]);
        }
    }
    return pts;
}

// "frame_2" < "frame_10": digit runs compare by value.
static bool naturalLess(const std::string &a, const std::string &b) {
    const auto dig = [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; };
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (dig(a[i]) && dig(b[j])) {
            size_t ie = i, je = j;
            while (ie < a.size() && dig(a[ie])) ++ie;
            while (je < b.size() && dig(b[je])) ++je;
            while (i + 1 < ie && a[i] == '0') ++i;  // leading zeros
            while (j + 1 < je && b[j] == '0') ++j;
            if (ie - i != je - j) return ie - i < je - j;
            const int c = a.compare(i, ie - i, b, j, je - j);
            if (c != 0) return c < 0;
            i = ie;
            j = je;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return a.size() - i < b.size() - j;
}

static std::vector<std::string> listPly(const fs::path &dir) {
    std::vector<std::string> files;
    for (const auto &e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".ply") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end(), naturalLess);
    return files;
}

// Rough alignment check: share of `src` points (moved by T) with a neighbour
// in `map` closer than `max_dist`.
static double inlierShare(const Cloud &src,
                          const Sophus::SE3d &T,
                          const kiss_icp::VoxelHashMap &map,
                          double max_dist) {
    if (src.empty()) return 0.0;
    size_t n = 0;
    for (const auto &p : src) {
        if (std::get<1>(map.GetClosestNeighbor(T * p)) < max_dist) ++n;
    }
    return static_cast<double>(n) / static_cast<double>(src.size());
}

static int usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m=100] "
                 "[-v voxel_m=max_range/100]\n",
                 argv0);
    return 1;
}

int main(int argc, char **argv) {
    std::vector<std::string> files;
    std::string out_path = "traj.txt";
    double max_range = 100.0, voxel = -1.0;

    for (int i = 1; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        if (!std::strcmp(argv[i], "-o") && has_val) {
            out_path = argv[++i];
        } else if (!std::strcmp(argv[i], "-r") && has_val) {
            max_range = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "-v") && has_val) {
            voxel = std::atof(argv[++i]);
        } else if (argv[i][0] == '-') {
            return usage(argv[0]);
        } else if (fs::is_directory(argv[i])) {
            const auto found = listPly(argv[i]);
            files.insert(files.end(), found.begin(), found.end());
        } else {
            files.emplace_back(argv[i]);
        }
    }
    if (voxel < 0.0) voxel = max_range / 100.0;
    if (max_range <= 0.0 || voxel <= 0.0) return usage(argv[0]);
    if (files.size() < 2) {
        std::fprintf(stderr, "need at least 2 PLY files (got %zu)\n", files.size());
        return usage(argv[0]);
    }

    FILE *out = std::fopen(out_path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }

    kiss_icp::pipeline::KISSConfig cfg;
    cfg.max_range = max_range;
    cfg.voxel_size = voxel;
    cfg.deskew = false;  // no per-point timestamps in the PLYs
    kiss_icp::pipeline::KissICP odom(cfg);
    const std::vector<double> no_stamps;

    std::printf("%zu files, max_range %.1f m, voxel %.2f m\n", files.size(), max_range, voxel);
    std::printf("%9s  %-28s %9s %9s %9s %7s %8s\n", "frame", "file", "points", "step[m]",
                "rot[deg]", "fit[%]", "ms");

    const double rad2deg = 180.0 / 3.14159265358979323846;
    const auto t_total = Clock::now();
    size_t n_ok = 0;
    double path_len = 0.0;

    for (size_t i = 0; i < files.size(); ++i) {
        const std::string name = fs::path(files[i]).filename().string();
        Cloud cloud;
        try {
            cloud = loadPly(files[i]);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "skip %s: %s\n", name.c_str(), e.what());
            continue;
        }
        if (cloud.empty()) {
            std::fprintf(stderr, "skip %s: no valid points\n", name.c_str());
            continue;
        }

        // Map as it was before this frame went in, for the fit check.
        const kiss_icp::VoxelHashMap map_prev = odom.VoxelMap();

        // First frame only seeds the map (pose = identity). Later frames are
        // registered against it, initial guess = constant-velocity prediction.
        const auto t0 = Clock::now();
        const Cloud src = std::get<1>(odom.RegisterFrame(cloud, no_stamps));
        const double ms = msSince(t0);

        const Sophus::SE3d &T = odom.pose();
        const Sophus::SE3d &d = odom.delta();
        const Eigen::Matrix<double, 3, 4> M = T.matrix3x4();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                std::fprintf(out, "%.9g%c", M(r, c), (r == 2 && c == 3) ? '\n' : ' ');
            }
        }

        const double step = d.translation().norm();
        path_len += step;
        std::printf("%4zu/%-4zu  %-28s %9zu ", i + 1, files.size(), name.c_str(), cloud.size());
        if (n_ok == 0) {
            std::printf("%9s %9s %7s %8.1f\n", "-", "-", "-", ms);
        } else {
            std::printf("%9.3f %9.3f %7.1f %8.1f\n", step, d.so3().log().norm() * rad2deg,
                        100.0 * inlierShare(src, T, map_prev, voxel), ms);
        }
        ++n_ok;
    }
    std::fclose(out);

    if (n_ok < 2) {
        std::fprintf(stderr, "fewer than 2 usable frames, no trajectory\n");
        return 1;
    }

    const Eigen::Matrix4d M = odom.pose().matrix();
    std::printf("\n%zu poses -> %s  (path %.2f m, %.1f s)\n", n_ok, out_path.c_str(), path_len,
                msSince(t_total) / 1000.0);
    std::printf("final pose T_0_n:\n");
    for (int r = 0; r < 4; ++r) {
        std::printf("  % .6f % .6f % .6f % .6f\n", M(r, 0), M(r, 1), M(r, 2), M(r, 3));
    }
    return 0;
}
