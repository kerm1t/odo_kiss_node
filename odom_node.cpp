// kiss_odom_node - eCAL 5 node: foxglove.PointCloud in -> KISS-ICP -> foxglove.Odometry out.
//
//   kiss_odom_node [--in pointcloud] [--out odometry] [--odom-frame odom] [--body-frame <id>]
//                  [-r max_range_m] [-v voxel_m] [-p max_points_per_voxel] [-i max_icp_iterations] [-q]
//
// Output per input cloud, stamped with the cloud's timestamp:
//   frame_id         odom frame (= sensor frame at the first cloud)
//   body_frame_id    the cloud's frame_id (or --body-frame)
//   pose             body in odom frame, absolute (accumulated)
//   linear/angular_velocity   per-epoch motion / dt, in the body frame
//   metadata         fit, sigma, proc_ms, dropped
//
// Registration runs on the main thread. The receive callback only parses the
// cloud into a single-slot buffer: if a new cloud arrives while the previous
// one is still being registered, the older unprocessed one is dropped.
// A timestamp going backwards (e.g. replay looping) resets the odometry.

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>

#include "foxglove/Odometry.pb.h"
#include "foxglove/PointCloud.pb.h"
#include "kiss_odom.hpp"

using Cloud = KissOdom::Cloud;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string in = "pointcloud";
    std::string out = "odometry";
    std::string odom_frame = "odom";
    std::string body_frame;  // empty: use the cloud's frame_id
    KissOdom::Params odom;
    double voxel = -1.0;  // default max_range / 100
    bool quiet = false;
};

// One received cloud, parsed, waiting for registration.
struct Pending {
    Cloud cloud;
    double stamp = -1.0;  // seconds
    google::protobuf::Timestamp ts;
    std::string frame_id;
    bool valid = false;
};

// Extract x/y/z (FLOAT32 or FLOAT64) from the packed data and move the points
// into frame_id if the message carries a non-identity pose.
static bool toCloud(const foxglove::PointCloud &msg, Cloud &out, std::string &err) {
    using Field = foxglove::PackedElementField;
    const Field *f[3] = {nullptr, nullptr, nullptr};
    for (const auto &field : msg.fields()) {
        if (field.name() == "x") f[0] = &field;
        if (field.name() == "y") f[1] = &field;
        if (field.name() == "z") f[2] = &field;
    }
    const size_t stride = msg.point_stride();
    const auto fail = [&err](const char *why) {
        err = why;
        return false;
    };
    if (!f[0] || !f[1] || !f[2]) return fail("needs x, y and z fields");
    if (stride == 0) return fail("point_stride is 0");
    size_t off[3];
    bool is_f32[3];
    for (int k = 0; k < 3; ++k) {
        if (f[k]->type() != Field::FLOAT32 && f[k]->type() != Field::FLOAT64) {
            return fail("x/y/z must be FLOAT32 or FLOAT64");
        }
        is_f32[k] = f[k]->type() == Field::FLOAT32;
        off[k] = f[k]->offset();
        if (off[k] + (is_f32[k] ? 4 : 8) > stride) return fail("field offset beyond point_stride");
    }

    const std::string &data = msg.data();
    const size_t n = data.size() / stride;
    out.clear();
    out.reserve(n);
    const char *p = data.data();
    for (size_t i = 0; i < n; ++i, p += stride) {
        double v[3];
        for (int k = 0; k < 3; ++k) {
            if (is_f32[k]) {
                float t;
                std::memcpy(&t, p + off[k], 4);
                v[k] = t;
            } else {
                std::memcpy(&v[k], p + off[k], 8);
            }
        }
        if (std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2])) {
            out.emplace_back(v[0], v[1], v[2]);
        }
    }

    // "pose" = origin of the cloud relative to frame_id. Unset or identity in the usual case.
    if (msg.has_pose()) {
        const auto &t = msg.pose().position();
        const auto &q = msg.pose().orientation();
        Eigen::Quaterniond rot(q.w(), q.x(), q.y(), q.z());
        if (rot.norm() < 1e-6) rot = Eigen::Quaterniond::Identity();  // unset orientation
        const Sophus::SE3d T(rot.normalized(), Eigen::Vector3d(t.x(), t.y(), t.z()));
        if (T.log().norm() > 1e-9) {
            for (auto &pt : out) pt = T * pt;
        }
    }
    return true;
}

static int usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--in pointcloud] [--out odometry] [--odom-frame odom] [--body-frame <id>]\n"
                 "       [-r max_range_m=100] [-v voxel_m=max_range/100] [-p max_points_per_voxel=20]\n"
                 "       [-i max_icp_iterations=500] [-q (no per-frame log)]\n",
                 argv0);
    return 1;
}

int main(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        const auto is = [&](const char *name) { return std::strcmp(argv[i], name) == 0; };
        if (is("--in") && has_val) {
            args.in = argv[++i];
        } else if (is("--out") && has_val) {
            args.out = argv[++i];
        } else if (is("--odom-frame") && has_val) {
            args.odom_frame = argv[++i];
        } else if (is("--body-frame") && has_val) {
            args.body_frame = argv[++i];
        } else if (is("-r") && has_val) {
            args.odom.max_range = std::atof(argv[++i]);
        } else if (is("-v") && has_val) {
            args.voxel = std::atof(argv[++i]);
        } else if (is("-p") && has_val) {
            args.odom.max_points_per_voxel = std::atoi(argv[++i]);
        } else if (is("-i") && has_val) {
            args.odom.max_iterations = std::atoi(argv[++i]);
        } else if (is("-q")) {
            args.quiet = true;
        } else {
            return usage(argv[0]);
        }
    }
    args.odom.voxel = args.voxel > 0.0 ? args.voxel : args.odom.max_range / 100.0;
    if (args.odom.max_range <= 0.0 || args.odom.max_points_per_voxel < 1 ||
        args.odom.max_iterations < 1) {
        return usage(argv[0]);
    }

    eCAL::Initialize(0, nullptr, "kiss_odom_node");
    eCAL::Process::SetState(proc_sev_healthy, proc_sev_level1, "waiting for point clouds");

    // Single-slot hand-over from the receive callback to the main loop.
    std::mutex mtx;
    std::condition_variable cv;
    Pending slot;
    size_t dropped = 0;

    eCAL::protobuf::CPublisher<foxglove::Odometry> pub(args.out);
    eCAL::protobuf::CSubscriber<foxglove::PointCloud> sub(args.in);

    sub.AddReceiveCallback([&](const char *, const foxglove::PointCloud &msg, long long send_time_us,
                               long long, long long) {
        Pending p;
        std::string err;
        if (!toCloud(msg, p.cloud, err)) {
            std::fprintf(stderr, "bad point cloud on '%s': %s\n", args.in.c_str(), err.c_str());
            return;
        }
        if (msg.has_timestamp()) {
            p.ts = msg.timestamp();
        } else {  // fall back to the eCAL send time
            p.ts.set_seconds(send_time_us / 1000000);
            p.ts.set_nanos(static_cast<int32_t>((send_time_us % 1000000) * 1000));
        }
        p.stamp = static_cast<double>(p.ts.seconds()) + 1e-9 * static_cast<double>(p.ts.nanos());
        p.frame_id = msg.frame_id();
        p.valid = true;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (slot.valid) ++dropped;  // previous cloud never got registered
            slot = std::move(p);
        }
        cv.notify_one();
    });

    std::printf("kiss_odom_node: '%s' (foxglove.PointCloud) -> '%s' (foxglove.Odometry)\n",
                args.in.c_str(), args.out.c_str());
    std::printf("max_range %.1f m, voxel %.2f m, %d pts/voxel, max %d ICP iterations\n",
                args.odom.max_range, args.odom.voxel, args.odom.max_points_per_voxel,
                args.odom.max_iterations);
    if (!args.quiet) {
        std::printf("%7s %8s %8s %8s %6s %7s %8s %7s %6s %7s %6s %7s\n", "frame", "points",
                    "step[m]", "rot[deg]", "fit[%]", "pre[ms]", "icp[ms]", "map[ms]", "src",
                    "vox", "sigma", "dropped");
    }
    std::fflush(stdout);

    KissOdom odom(args.odom);
    const double rad2deg = 180.0 / 3.14159265358979323846;
    size_t n_frames = 0;
    double last_stamp = -1.0;

    while (eCAL::Ok()) {
        Pending p;
        size_t n_dropped = 0;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return slot.valid; });
            if (!slot.valid) continue;
            p = std::move(slot);
            slot.valid = false;
            n_dropped = dropped;
        }
        if (p.cloud.empty()) {
            std::fprintf(stderr, "empty point cloud, skipped\n");
            continue;
        }
        if (last_stamp >= 0.0 && p.stamp < last_stamp) {
            std::printf("timestamp went backwards (%.3f -> %.3f): odometry reset\n", last_stamp,
                        p.stamp);
            std::fflush(stdout);
            odom.reset();
        }
        last_stamp = p.stamp;

        const auto t0 = Clock::now();
        const KissOdom::Result r = odom.registerFrame(p.cloud, p.stamp);
        const double proc_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        ++n_frames;

        foxglove::Odometry msg;
        *msg.mutable_timestamp() = p.ts;
        msg.set_frame_id(args.odom_frame);
        msg.set_body_frame_id(args.body_frame.empty() ? p.frame_id : args.body_frame);
        const Eigen::Vector3d t = r.pose.translation();
        const Eigen::Quaterniond q = r.pose.unit_quaternion();
        auto *pos = msg.mutable_pose()->mutable_position();
        pos->set_x(t.x());
        pos->set_y(t.y());
        pos->set_z(t.z());
        auto *ori = msg.mutable_pose()->mutable_orientation();
        ori->set_x(q.x());
        ori->set_y(q.y());
        ori->set_z(q.z());
        ori->set_w(q.w());
        if (!r.first && r.dt > 0.0) {  // body-frame twist over the last epoch
            const Eigen::Matrix<double, 6, 1> twist = r.delta.log() / r.dt;
            auto *lin = msg.mutable_linear_velocity();
            lin->set_x(twist[0]);
            lin->set_y(twist[1]);
            lin->set_z(twist[2]);
            auto *ang = msg.mutable_angular_velocity();
            ang->set_x(twist[3]);
            ang->set_y(twist[4]);
            ang->set_z(twist[5]);
        }
        msg.mutable_pose_covariance()->Resize(36, 0.0);  // unknown
        msg.mutable_velocity_covariance()->Resize(36, 0.0);
        const auto meta = [&msg](const char *key, const std::string &value) {
            auto *kv = msg.add_metadata();
            kv->set_key(key);
            kv->set_value(value);
        };
        meta("fit", std::to_string(r.fit));
        meta("sigma", std::to_string(r.sigma));
        meta("proc_ms", std::to_string(proc_ms));
        meta("dropped", std::to_string(n_dropped));
        pub.Send(msg);

        // Process state for the eCAL monitor: warn when the registration looks poor.
        char info[128];
        std::snprintf(info, sizeof(info), "frame %zu, fit %.0f %%, %.0f ms, dropped %zu", n_frames,
                      100.0 * r.fit, proc_ms, n_dropped);
        eCAL::Process::SetState(r.first || r.fit >= 0.5 ? proc_sev_healthy : proc_sev_warning,
                                proc_sev_level1, info);

        if (!args.quiet) {
            std::printf("%7zu %8zu ", n_frames, p.cloud.size());
            if (r.first) {
                std::printf("%8s %8s %6s ", "-", "-", "-");
            } else {
                std::printf("%8.3f %8.3f %6.1f ", r.delta.translation().norm(),
                            r.delta.so3().log().norm() * rad2deg, 100.0 * r.fit);
            }
            std::printf("%7.1f %8.1f %7.1f %6zu %7zu %6.2f %7zu\n", r.ms_pre, r.ms_icp, r.ms_map,
                        r.n_src, r.n_voxels, r.sigma, n_dropped);
            std::fflush(stdout);
        }
    }

    eCAL::Finalize();
    return 0;
}
