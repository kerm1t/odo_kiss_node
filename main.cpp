// ply_odom - KISS-ICP odometry over a sequence of PLY scans, with live viewer.
//
//   ply_odom <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m] [-v voxel_m]
//            [-d vis_voxel_m] [-n]
//
// A directory is scanned for *.ply and sorted naturally (frame_2 < frame_10);
// files given explicitly keep their order. Writes one pose per frame in KITTI
// format (3x4 row-major, 12 values per line). Poses are T_0_i with
// p_0 = T_0_i * p_i, i.e. the sensor pose at frame i in the first frame.
//
// Viewer (SDL2 + OpenGL 3.3, see viewer.hpp for controls) shows the accumulated
// cloud in the first frame's coordinates plus the ego trajectory. -n disables it.

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <happly.h>
#include <kiss_icp/core/VoxelUtils.hpp>
#include <kiss_icp/pipeline/KissICP.hpp>

#ifdef PLY_ODOM_VIEWER
#include "viewer.hpp"
#endif

namespace fs = std::filesystem;
using Cloud = std::vector<Eigen::Vector3d>;
using Clock = std::chrono::steady_clock;

struct Options {
    std::vector<std::string> files;
    std::string out_path = "traj.txt";
    double max_range = 100.0;
    double voxel = -1.0;      // default max_range / 100
    double vis_voxel = -1.0;  // default voxel / 4
    bool viewer = true;
};

// Called once per registered frame: range-cropped scan (sensor frame), its
// pose in the first frame, number of frames done, number of input files.
using FrameSink = std::function<void(const Cloud &, const Sophus::SE3d &, size_t, size_t)>;

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

// Runs the whole sequence, writes the trajectory, prints the log.
// Returns the number of poses written.
static size_t runOdometry(const Options &opt, const FrameSink &sink, const std::atomic<bool> &stop) {
    FILE *out = std::fopen(opt.out_path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", opt.out_path.c_str());
        return 0;
    }

    kiss_icp::pipeline::KISSConfig cfg;
    cfg.max_range = opt.max_range;
    cfg.voxel_size = opt.voxel;
    cfg.deskew = false;  // no per-point timestamps in the PLYs
    kiss_icp::pipeline::KissICP odom(cfg);
    const std::vector<double> no_stamps;
    const auto &files = opt.files;

    std::printf("%zu files, max_range %.1f m, voxel %.2f m\n", files.size(), opt.max_range,
                opt.voxel);
    // odom[ms] = KISS-ICP RegisterFrame only, frame[ms] = whole iteration
    // (load, registration, fit check, viewer accumulation).
    std::printf("%9s  %-28s %9s %9s %9s %7s %9s %9s\n", "frame", "file", "points", "step[m]",
                "rot[deg]", "fit[%]", "odom[ms]", "frame[ms]");

    const double rad2deg = 180.0 / 3.14159265358979323846;
    const auto t_total = Clock::now();
    size_t n_ok = 0;
    double path_len = 0.0;

    for (size_t i = 0; i < files.size() && !stop; ++i) {
        const auto t_frame = Clock::now();
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
        const auto [frame, src] = odom.RegisterFrame(cloud, no_stamps);
        const double ms = msSince(t0);

        const Sophus::SE3d T = odom.pose();
        const Sophus::SE3d d = odom.delta();
        const Eigen::Matrix<double, 3, 4> M = T.matrix3x4();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                std::fprintf(out, "%.9g%c", M(r, c), (r == 2 && c == 3) ? '\n' : ' ');
            }
        }
        std::fflush(out);

        const double step = d.translation().norm();
        const double fit = n_ok == 0 ? 0.0 : 100.0 * inlierShare(src, T, map_prev, opt.voxel);
        path_len += step;
        if (sink) sink(frame, T, i + 1, files.size());

        std::printf("%4zu/%-4zu  %-28s %9zu ", i + 1, files.size(), name.c_str(), cloud.size());
        if (n_ok == 0) {
            std::printf("%9s %9s %7s %9.1f %9.1f\n", "-", "-", "-", ms, msSince(t_frame));
        } else {
            std::printf("%9.3f %9.3f %7.1f %9.1f %9.1f\n", step, d.so3().log().norm() * rad2deg,
                        fit, ms, msSince(t_frame));
        }
        std::fflush(stdout);
        ++n_ok;
    }
    std::fclose(out);

    if (stop) std::printf("\nstopped early\n");
    if (n_ok < 2) {
        std::fprintf(stderr, "fewer than 2 usable frames, no trajectory\n");
        return n_ok;
    }
    const Eigen::Matrix4d M = odom.pose().matrix();
    std::printf("\n%zu poses -> %s  (path %.2f m, %.1f s)\n", n_ok, opt.out_path.c_str(), path_len,
                msSince(t_total) / 1000.0);
    std::printf("final pose T_0_n:\n");
    for (int r = 0; r < 4; ++r) {
        std::printf("  % .6f % .6f % .6f % .6f\n", M(r, 0), M(r, 1), M(r, 2), M(r, 3));
    }
    std::fflush(stdout);
    return n_ok;
}

#ifdef PLY_ODOM_VIEWER
// Accumulates scans for display: one point per vis voxel, world-wide, so the
// buffer grows with covered area and not with the number of frames.
// Occupancy is kept as one bit per voxel in 8x8x8 blocks (64 bytes per block),
// which costs a few bytes per stored point instead of ~40 for a hash set of keys.
class VisAccumulator {
public:
    explicit VisAccumulator(double voxel) : voxel_(voxel) {}

    // Returns the newly seen points as x y z s (world xyz, s = height in the
    // sensor frame, used for colouring).
    std::vector<float> add(const Cloud &frame, const Sophus::SE3d &T) {
        std::vector<float> out;
        for (const auto &p : kiss_icp::VoxelDownsample(frame, voxel_)) {
            const Eigen::Vector3d w = T * p;
            if (!markNew(w)) continue;
            out.insert(out.end(), {static_cast<float>(w.x()), static_cast<float>(w.y()),
                                   static_cast<float>(w.z()), static_cast<float>(p.z())});
        }
        return out;
    }

private:
    // Sets the voxel's bit; true if it was not set before.
    bool markNew(const Eigen::Vector3d &w) {
        const int64_t ix = static_cast<int64_t>(std::floor(w.x() / voxel_));
        const int64_t iy = static_cast<int64_t>(std::floor(w.y() / voxel_));
        const int64_t iz = static_cast<int64_t>(std::floor(w.z() / voxel_));
        const auto q = [](int64_t v) { return static_cast<uint64_t>((v >> 3) + (1 << 20)) & 0x1FFFFF; };
        const uint64_t block = (q(ix) << 42) | (q(iy) << 21) | q(iz);
        const unsigned bit = static_cast<unsigned>(((ix & 7) << 6) | ((iy & 7) << 3) | (iz & 7));
        uint64_t &word = blocks_[block][bit >> 6];  // new blocks start zeroed
        const uint64_t mask = uint64_t{1} << (bit & 63);
        if (word & mask) return false;
        word |= mask;
        return true;
    }

    double voxel_;
    std::unordered_map<uint64_t, std::array<uint64_t, 8>> blocks_;
};

// Odometry runs in a worker thread, the viewer on the main thread.
static int runWithViewer(const Options &opt, Viewer &viewer) {
    struct {
        std::mutex m;
        std::vector<float> pts;
        std::vector<Eigen::Matrix4f> poses;
        size_t done = 0;
        bool finished = false;
    } shared;

    VisAccumulator acc(opt.vis_voxel);
    std::atomic<bool> stop{false};
    size_t n_ok = 0;

    std::thread worker([&] {
        n_ok = runOdometry(
            opt,
            [&](const Cloud &frame, const Sophus::SE3d &T, size_t done, size_t) {
                const std::vector<float> fresh = acc.add(frame, T);
                std::lock_guard<std::mutex> lock(shared.m);
                shared.pts.insert(shared.pts.end(), fresh.begin(), fresh.end());
                shared.poses.push_back(T.matrix().cast<float>());
                shared.done = done;
            },
            stop);
        std::lock_guard<std::mutex> lock(shared.m);
        shared.finished = true;
    });

    bool finished = false;
    while (viewer.frame()) {
        std::vector<float> pts;
        std::vector<Eigen::Matrix4f> poses;
        size_t done = 0;
        bool fin = false;
        {
            std::lock_guard<std::mutex> lock(shared.m);
            pts.swap(shared.pts);
            poses.swap(shared.poses);
            done = shared.done;
            fin = shared.finished;
        }
        if (poses.empty() && fin == finished) continue;
        finished = fin;
        viewer.addPoints(pts);
        for (const auto &T : poses) viewer.addPose(T);
        char status[128];
        std::snprintf(status, sizeof(status), "%zu/%zu frames%s  |  %.2f M pts", done,
                      opt.files.size(), fin ? " (done)" : "",
                      static_cast<double>(viewer.numPoints()) / 1.0e6);
        viewer.setStatus(status);
    }
    stop = true;  // window closed: let the current frame finish, then stop
    worker.join();
    return n_ok >= 2 ? 0 : 1;
}
#endif

static int usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m=100]\n"
                 "       [-v voxel_m=max_range/100] [-d vis_voxel_m=voxel/4] [-n (no viewer)]\n",
                 argv0);
    return 1;
}

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        if (!std::strcmp(argv[i], "-o") && has_val) {
            opt.out_path = argv[++i];
        } else if (!std::strcmp(argv[i], "-r") && has_val) {
            opt.max_range = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "-v") && has_val) {
            opt.voxel = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "-d") && has_val) {
            opt.vis_voxel = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "-n")) {
            opt.viewer = false;
        } else if (argv[i][0] == '-') {
            return usage(argv[0]);
        } else if (fs::is_directory(argv[i])) {
            const auto found = listPly(argv[i]);
            opt.files.insert(opt.files.end(), found.begin(), found.end());
        } else {
            opt.files.emplace_back(argv[i]);
        }
    }
    if (opt.voxel < 0.0) opt.voxel = opt.max_range / 100.0;
    if (opt.vis_voxel < 0.0) opt.vis_voxel = opt.voxel / 4.0;
    if (opt.max_range <= 0.0 || opt.voxel <= 0.0 || opt.vis_voxel <= 0.0) return usage(argv[0]);
    if (opt.files.size() < 2) {
        std::fprintf(stderr, "need at least 2 PLY files (got %zu)\n", opt.files.size());
        return usage(argv[0]);
    }

#ifdef PLY_ODOM_VIEWER
    if (opt.viewer) {
        Viewer viewer;
        if (viewer.init("ply_odom")) return runWithViewer(opt, viewer);
        std::fprintf(stderr, "no viewer, running headless\n");
    }
#endif
    const std::atomic<bool> never_stop{false};
    return runOdometry(opt, nullptr, never_stop) >= 2 ? 0 : 1;
}
