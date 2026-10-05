// vis_feed.hpp - hand-over of registered scans from the odometry thread to the
// viewer on the main thread. Used by the PLY app and the eCAL node.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <kiss_icp/core/VoxelUtils.hpp>
#include <sophus/se3.hpp>

#include "viewer.hpp"

// Accumulates scans for display: one point per vis voxel, world-wide, so the
// buffer grows with covered area and not with the number of frames.
// Occupancy is kept as one bit per voxel in 8x8x8 blocks (64 bytes per block),
// which costs a few bytes per stored point instead of ~40 for a hash set of keys.
class VisAccumulator {
public:
    using Cloud = std::vector<Eigen::Vector3d>;

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

    void clear() { blocks_.clear(); }

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

// push / setStatus / reset are called from the odometry thread, drain from the
// thread that owns the viewer.
class VisFeed {
public:
    using Cloud = VisAccumulator::Cloud;

    explicit VisFeed(double vis_voxel) : acc_(vis_voxel) {}

    // A registered scan (sensor frame) with its pose in the odometry frame.
    // `stamp` in seconds, < 0 if unknown (then the viewer shows no velocity).
    void push(const Cloud &frame, const Sophus::SE3d &T, double stamp, const std::string &status) {
        if (closed()) return;
        const std::vector<float> fresh = acc_.add(frame, T);
        std::lock_guard<std::mutex> lock(m_);
        pts_.insert(pts_.end(), fresh.begin(), fresh.end());
        poses_.emplace_back(T.matrix().cast<float>(), stamp);
        status_ = status;
        dirty_ = true;
    }

    void setStatus(const std::string &status) {
        std::lock_guard<std::mutex> lock(m_);
        status_ = status;
        dirty_ = true;
    }

    // Odometry started over: drop everything shown so far.
    void reset() {
        acc_.clear();
        std::lock_guard<std::mutex> lock(m_);
        pts_.clear();
        poses_.clear();
        reset_ = true;
        dirty_ = true;
    }

    // Viewer is gone: ignore further data.
    void close() {
        std::lock_guard<std::mutex> lock(m_);
        closed_ = true;
        pts_.clear();
        poses_.clear();
    }

    // Move whatever arrived into the viewer and update its title.
    void drain(Viewer &viewer) {
        std::vector<float> pts;
        std::vector<std::pair<Eigen::Matrix4f, double>> poses;
        std::string status;
        bool reset = false;
        {
            std::lock_guard<std::mutex> lock(m_);
            if (!dirty_) return;
            pts.swap(pts_);
            poses.swap(poses_);
            status = status_;
            reset = reset_;
            reset_ = dirty_ = false;
        }
        if (reset) viewer.clear();
        viewer.addPoints(pts);
        for (const auto &p : poses) viewer.addPose(p.first, p.second);
        char n_pts[48];
        std::snprintf(n_pts, sizeof(n_pts), "  |  %.2f M pts",
                      static_cast<double>(viewer.numPoints()) / 1.0e6);
        viewer.setStatus(status + n_pts);
    }

private:
    bool closed() {
        std::lock_guard<std::mutex> lock(m_);
        return closed_;
    }

    VisAccumulator acc_;  // odometry thread only
    std::mutex m_;
    std::vector<float> pts_;
    std::vector<std::pair<Eigen::Matrix4f, double>> poses_;
    std::string status_;
    bool dirty_ = false, reset_ = false, closed_ = false;
};
