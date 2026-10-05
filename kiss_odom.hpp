// kiss_odom.hpp - KISS-ICP odometry step, shared by the PLY app and the eCAL node.
//
// Same steps and defaults as kiss_icp::pipeline::KissICP::RegisterFrame (v1.3.0),
// written out on the core modules so every stage can be timed and the fit can
// be checked before the scan goes into the map.
//
// One addition: if frames carry timestamps, the constant-velocity prediction
// is scaled by the ratio of the time steps, so a dropped or late frame does
// not throw the initial guess off. With regular steps (within 5 %) or without
// timestamps the behaviour is exactly stock KISS-ICP.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <kiss_icp/core/Preprocessing.hpp>
#include <kiss_icp/core/Registration.hpp>
#include <kiss_icp/core/Threshold.hpp>
#include <kiss_icp/core/VoxelHashMap.hpp>
#include <kiss_icp/core/VoxelUtils.hpp>
#include <kiss_icp/pipeline/KissICP.hpp>  // KISSConfig defaults
#include <sophus/se3.hpp>

class KissOdom {
public:
    using Cloud = std::vector<Eigen::Vector3d>;

    struct Params {
        double max_range = 100.0;
        double voxel = 1.0;
        int max_points_per_voxel = kiss_icp::pipeline::KISSConfig().max_points_per_voxel;
        int max_iterations = kiss_icp::pipeline::KISSConfig().max_num_iterations;
    };

    struct Result {
        Sophus::SE3d pose;   // T_0_i: sensor at frame i in the first frame, p_0 = T_0_i * p_i
        Sophus::SE3d delta;  // T_(i-1)_i: step since the previous frame
        double dt = -1.0;    // time since the previous frame [s], < 0 if unknown
        bool first = false;  // first frame: only seeds the map, pose = identity
        double fit = 0.0;    // share of ICP points with a map neighbour < voxel, 0..1
        double ms_pre = 0.0, ms_icp = 0.0, ms_map = 0.0;  // crop+downsample, ICP, map update
        size_t n_src = 0;     // points used for ICP
        size_t n_voxels = 0;  // voxels in the local map
        double sigma = 0.0;   // adaptive threshold used for this frame
        Cloud frame;          // range-cropped input scan (sensor frame)
    };

    explicit KissOdom(const Params &p)
        : cfg_(makeConfig(p)),
          preprocessor_(cfg_.max_range, cfg_.min_range, cfg_.deskew, cfg_.max_num_threads),
          registration_(cfg_.max_num_iterations, cfg_.convergence_criterion, cfg_.max_num_threads),
          map_(cfg_.voxel_size, cfg_.max_range, static_cast<unsigned int>(cfg_.max_points_per_voxel)),
          threshold_(cfg_.initial_threshold, cfg_.min_motion_th, cfg_.max_range) {}

    const kiss_icp::pipeline::KISSConfig &config() const { return cfg_; }

    // Forget map, pose and motion model.
    void reset() {
        map_.Clear();
        threshold_ = kiss_icp::AdaptiveThreshold(cfg_.initial_threshold, cfg_.min_motion_th,
                                                 cfg_.max_range);
        pose_ = Sophus::SE3d();
        delta_ = Sophus::SE3d();
        last_stamp_ = last_dt_ = -1.0;
        first_ = true;
    }

    // Register one scan (sensor frame). `stamp` in seconds, < 0 if unknown.
    Result registerFrame(const Cloud &cloud, double stamp = -1.0) {
        using Clock = std::chrono::steady_clock;
        const auto ms = [](Clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        };
        Result r;
        r.first = first_;

        // Range crop, then two-stage downsampling: `frame_ds` goes into the map,
        // the sparser `src` is what ICP aligns.
        auto t0 = Clock::now();
        static const std::vector<double> no_stamps;  // no per-point times, no deskew
        r.frame = preprocessor_.Preprocess(cloud, no_stamps, delta_);
        const Cloud frame_ds = kiss_icp::VoxelDownsample(r.frame, cfg_.voxel_size * 0.5);
        const Cloud src = kiss_icp::VoxelDownsample(frame_ds, cfg_.voxel_size * 1.5);
        r.ms_pre = ms(t0);
        r.n_src = src.size();

        // Constant-velocity prediction, scaled if the time step changed.
        Sophus::SE3d predicted = delta_;
        if (stamp >= 0.0 && last_stamp_ >= 0.0) {
            r.dt = stamp - last_stamp_;
            if (r.dt > 0.0 && last_dt_ > 0.0) {
                const double ratio = std::min(r.dt / last_dt_, 3.0);
                if (std::abs(ratio - 1.0) > 0.05) {
                    predicted = Sophus::SE3d::exp(ratio * delta_.log());
                }
            }
        }

        // ICP against the local map. First frame: empty map, pose stays identity.
        t0 = Clock::now();
        r.sigma = threshold_.ComputeThreshold();
        const Sophus::SE3d guess = pose_ * predicted;
        const Sophus::SE3d T = registration_.AlignPointsToMap(src, map_, guess, 3.0 * r.sigma, r.sigma);
        r.ms_icp = ms(t0);

        if (!first_ && !src.empty()) {
            size_t n = 0;
            for (const auto &p : src) {
                if (std::get<1>(map_.GetClosestNeighbor(T * p)) < cfg_.voxel_size) ++n;
            }
            r.fit = static_cast<double>(n) / static_cast<double>(src.size());
        }

        t0 = Clock::now();
        threshold_.UpdateModelDeviation(guess.inverse() * T);
        map_.Update(frame_ds, T);
        r.ms_map = ms(t0);
        r.n_voxels = map_.map_.size();

        delta_ = pose_.inverse() * T;
        pose_ = T;
        if (stamp >= 0.0) {
            last_dt_ = r.dt;
            last_stamp_ = stamp;
        }
        first_ = false;

        r.pose = pose_;
        r.delta = delta_;
        return r;
    }

private:
    static kiss_icp::pipeline::KISSConfig makeConfig(const Params &p) {
        kiss_icp::pipeline::KISSConfig c;
        c.max_range = p.max_range;
        c.voxel_size = p.voxel;
        c.max_points_per_voxel = p.max_points_per_voxel;
        c.max_num_iterations = p.max_iterations;
        c.deskew = false;
        return c;
    }

    kiss_icp::pipeline::KISSConfig cfg_;
    kiss_icp::Preprocessor preprocessor_;
    kiss_icp::Registration registration_;
    kiss_icp::VoxelHashMap map_;
    kiss_icp::AdaptiveThreshold threshold_;
    Sophus::SE3d pose_, delta_;
    double last_stamp_ = -1.0, last_dt_ = -1.0;
    bool first_ = true;
};
