// ply_odom - KISS-ICP odometry over a sequence of PLY scans, with live viewer.
//
//   ply_odom <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m] [-v voxel_m]
//            [-p max_points_per_voxel] [-i max_icp_iterations] [-d vis_voxel_m] [-f frame_rate_hz] [-n]
//
// A directory is scanned for *.ply and sorted naturally (frame_2 < frame_10);
// files given explicitly keep their order. Writes one pose per frame in KITTI
// format (3x4 row-major, 12 values per line). Poses are T_0_i with
// p_0 = T_0_i * p_i, i.e. the sensor pose at frame i in the first frame.
//
// Viewer (SDL2 + OpenGL 3.3, see viewer.hpp for controls) shows the accumulated
// cloud in the first frame's coordinates plus the ego trajectory. -n disables it.
// PLY files carry no time, so velocity and acceleration in the viewer need the
// frame rate of the recording: -f <hz>.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "kiss_odom.hpp"
#include "ply_io.hpp"

#ifdef PLY_ODOM_VIEWER
#include "vis_feed.hpp"
#endif

namespace fs = std::filesystem;
using Cloud = KissOdom::Cloud;
using Clock = std::chrono::steady_clock;

struct Options {
    std::vector<std::string> files;
    std::string out_path = "traj.txt";
    double max_range = 100.0;
    double voxel = -1.0;      // default max_range / 100
    double vis_voxel = -1.0;  // default voxel / 4
    int max_points_per_voxel = KissOdom::Params().max_points_per_voxel;
    int max_iterations = KissOdom::Params().max_iterations;
    double frame_hz = -1.0;   // frame rate of the recording, < 0 if unknown
    bool viewer = true;
};

// Called once per registered frame: range-cropped scan (sensor frame), its
// pose in the first frame, its time in seconds (< 0 if unknown), number of
// frames done.
using FrameSink = std::function<void(const Cloud &, const Sophus::SE3d &, double, size_t)>;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Runs the whole sequence, writes the trajectory, prints the log.
// Returns the number of poses written.
static size_t runOdometry(const Options &opt, const FrameSink &sink, const std::atomic<bool> &stop) {
    FILE *out = std::fopen(opt.out_path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", opt.out_path.c_str());
        return 0;
    }

    KissOdom::Params params;
    params.max_range = opt.max_range;
    params.voxel = opt.voxel;
    params.max_points_per_voxel = opt.max_points_per_voxel;
    params.max_iterations = opt.max_iterations;
    KissOdom odom(params);
    const auto &files = opt.files;

    std::printf("%zu files, max_range %.1f m, voxel %.2f m, %d pts/voxel, max %d ICP iterations\n",
                files.size(), params.max_range, params.voxel, params.max_points_per_voxel,
                params.max_iterations);
    // pre = range crop + downsampling, icp = registration, map = local map update,
    // frame = whole iteration incl. PLY load, fit check and viewer accumulation.
    // src = points used for ICP, vox = voxels in the local map, sigma = adaptive threshold.
    std::printf("%9s  %-22s %8s %8s %8s %6s %7s %8s %7s %8s %6s %7s %6s\n", "frame", "file",
                "points", "step[m]", "rot[deg]", "fit[%]", "pre[ms]", "icp[ms]", "map[ms]",
                "frame[ms]", "src", "vox", "sigma");

    const double rad2deg = 180.0 / 3.14159265358979323846;
    const auto t_total = Clock::now();
    size_t n_ok = 0;
    double path_len = 0.0;
    Sophus::SE3d pose;

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

        // First frame only seeds the map (pose = identity). Later frames are
        // registered against it, initial guess = constant-velocity prediction.
        const double stamp = opt.frame_hz > 0.0 ? static_cast<double>(i) / opt.frame_hz : -1.0;
        const KissOdom::Result r = odom.registerFrame(cloud, stamp);
        pose = r.pose;

        const Eigen::Matrix<double, 3, 4> M = r.pose.matrix3x4();
        for (int row = 0; row < 3; ++row) {
            for (int c = 0; c < 4; ++c) {
                std::fprintf(out, "%.9g%c", M(row, c), (row == 2 && c == 3) ? '\n' : ' ');
            }
        }
        std::fflush(out);

        const double step = r.delta.translation().norm();
        path_len += step;
        if (sink) sink(r.frame, r.pose, stamp, i + 1);

        std::printf("%4zu/%-4zu  %-22s %8zu ", i + 1, files.size(), name.c_str(), cloud.size());
        if (r.first) {
            std::printf("%8s %8s %6s ", "-", "-", "-");
        } else {
            std::printf("%8.3f %8.3f %6.1f ", step, r.delta.so3().log().norm() * rad2deg,
                        100.0 * r.fit);
        }
        std::printf("%7.1f %8.1f %7.1f %8.1f %6zu %7zu %6.2f\n", r.ms_pre, r.ms_icp, r.ms_map,
                    msSince(t_frame), r.n_src, r.n_voxels, r.sigma);
        std::fflush(stdout);
        ++n_ok;
    }
    std::fclose(out);

    if (stop) std::printf("\nstopped early\n");
    if (n_ok < 2) {
        std::fprintf(stderr, "fewer than 2 usable frames, no trajectory\n");
        return n_ok;
    }
    const Eigen::Matrix4d M = pose.matrix();
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
// Odometry runs in a worker thread, the viewer on the main thread.
static int runWithViewer(const Options &opt, Viewer &viewer) {
    VisFeed feed(opt.vis_voxel);
    std::atomic<bool> stop{false};
    size_t n_ok = 0;
    const auto status = [&opt](size_t done, bool finished) {
        char s[96];
        std::snprintf(s, sizeof(s), "%zu/%zu frames%s", done, opt.files.size(),
                      finished ? " (done)" : "");
        return std::string(s);
    };

    std::thread worker([&] {
        size_t last = 0;
        n_ok = runOdometry(
            opt,
            [&](const Cloud &frame, const Sophus::SE3d &T, double stamp, size_t done) {
                feed.push(frame, T, stamp, status(done, false));
                last = done;
            },
            stop);
        feed.setStatus(status(last, true));
    });

    while (viewer.frame()) feed.drain(viewer);
    stop = true;  // window closed: let the current frame finish, then stop
    worker.join();
    return n_ok >= 2 ? 0 : 1;
}
#endif

static int usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m=100]\n"
                 "       [-v voxel_m=max_range/100] [-p max_points_per_voxel=20]\n"
                 "       [-i max_icp_iterations=500] [-d vis_voxel_m=voxel/4] [-n (no viewer)]\n"
                 "       [-f frame_rate_hz (for velocity in the viewer)]\n",
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
        } else if (!std::strcmp(argv[i], "-p") && has_val) {
            opt.max_points_per_voxel = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-i") && has_val) {
            opt.max_iterations = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-f") && has_val) {
            opt.frame_hz = std::atof(argv[++i]);
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
    if (opt.max_range <= 0.0 || opt.voxel <= 0.0 || opt.vis_voxel <= 0.0 ||
        opt.max_points_per_voxel < 1 || opt.max_iterations < 1) {
        return usage(argv[0]);
    }
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
