// ply_odom - relative pose between two PLY scans via KISS-ICP.
//
//   ply_odom <a.ply> <b.ply> [max_range_m=100] [voxel_m=max_range/100]
//
// Prints T_a_b with p_a = T_a_b * p_b, i.e. the sensor pose at scan b
// expressed in scan a's frame (= the odometry step a -> b).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <happly.h>
#include <kiss_icp/pipeline/KissICP.hpp>

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

// Rough alignment check: share of `src` points (moved by T) that have a
// neighbour in `map` closer than `max_dist`, plus the RMSE over those.
struct Fit {
    double inliers = 0.0;
    double rmse = 0.0;
};

static Fit evalFit(const Cloud &src,
                   const Sophus::SE3d &T,
                   const kiss_icp::VoxelHashMap &map,
                   double max_dist) {
    size_t n = 0;
    double sq = 0.0;
    for (const auto &p : src) {
        const double d = std::get<1>(map.GetClosestNeighbor(T * p));
        if (d < max_dist) {
            ++n;
            sq += d * d;
        }
    }
    Fit f;
    if (!src.empty()) f.inliers = static_cast<double>(n) / static_cast<double>(src.size());
    if (n > 0) f.rmse = std::sqrt(sq / static_cast<double>(n));
    return f;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <a.ply> <b.ply> [max_range_m=100] [voxel_m=max_range/100]\n",
                     argv[0]);
        return 1;
    }

    kiss_icp::pipeline::KISSConfig cfg;
    cfg.max_range = argc > 3 ? std::atof(argv[3]) : 100.0;
    cfg.voxel_size = argc > 4 ? std::atof(argv[4]) : cfg.max_range / 100.0;
    cfg.deskew = false;  // single scans, no per-point timestamps
    if (cfg.max_range <= 0.0 || cfg.voxel_size <= 0.0) {
        std::fprintf(stderr, "max_range and voxel must be > 0\n");
        return 1;
    }

    Cloud a, b;
    try {
        auto t0 = Clock::now();
        a = loadPly(argv[1]);
        b = loadPly(argv[2]);
        std::printf("loaded  a: %zu pts, b: %zu pts  (%.0f ms)\n", a.size(), b.size(), msSince(t0));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "PLY load failed: %s\n", e.what());
        return 1;
    }
    if (a.empty() || b.empty()) {
        std::fprintf(stderr, "empty cloud\n");
        return 1;
    }

    kiss_icp::pipeline::KissICP odom(cfg);
    const std::vector<double> no_stamps;

    // Scan a only seeds the local map; pose stays identity.
    auto t0 = Clock::now();
    odom.RegisterFrame(a, no_stamps);
    const kiss_icp::VoxelHashMap map_a = odom.VoxelMap();  // keep a-only map for the fit check
    const double ms_a = msSince(t0);

    // Scan b is registered against that map, initial guess = identity.
    t0 = Clock::now();
    const Cloud src_b = std::get<1>(odom.RegisterFrame(b, no_stamps));  // downsampled b
    const double ms_b = msSince(t0);

    const Sophus::SE3d T = odom.pose();
    const Eigen::Matrix4d M = T.matrix();
    const Eigen::Vector3d t = T.translation();
    const double rad2deg = 180.0 / 3.14159265358979323846;
    const Eigen::Vector3d rv = T.so3().log() * rad2deg;  // rotation vector [deg]

    const Fit before = evalFit(src_b, Sophus::SE3d(), map_a, cfg.voxel_size);
    const Fit after = evalFit(src_b, T, map_a, cfg.voxel_size);

    std::printf("config  max_range %.1f m, voxel %.2f m, %zu src pts after downsampling\n",
                cfg.max_range, cfg.voxel_size, src_b.size());
    std::printf("timing  map init %.1f ms, registration %.1f ms\n\n", ms_a, ms_b);

    std::printf("T_a_b (p_a = T_a_b * p_b):\n");
    for (int r = 0; r < 4; ++r) {
        std::printf("  % .6f % .6f % .6f % .6f\n", M(r, 0), M(r, 1), M(r, 2), M(r, 3));
    }
    std::printf("\ntrans [m]    % .4f % .4f % .4f   |t| = %.4f\n", t.x(), t.y(), t.z(), t.norm());
    std::printf("rotvec [deg] % .4f % .4f % .4f   angle = %.4f\n", rv.x(), rv.y(), rv.z(),
                rv.norm());
    std::printf("\nfit (NN < %.2f m)  identity: %5.1f %% inliers, rmse %.3f m\n", cfg.voxel_size,
                100.0 * before.inliers, before.rmse);
    std::printf("                    aligned:  %5.1f %% inliers, rmse %.3f m\n",
                100.0 * after.inliers, after.rmse);
    return 0;
}
