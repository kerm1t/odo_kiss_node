// kiss_odom_node - eCAL 5 node: foxglove.PointCloud in -> KISS-ICP -> foxglove.Odometry out.
//
//   kiss_odom_node [--in pointcloud] [--out odometry] [--odom-frame odom] [--body-frame <id>]
//                  [-r max_range_m] [-v voxel_m] [-p max_points_per_voxel] [-i max_icp_iterations] [-q]
//                  [--viz] [-d vis_voxel_m] [--time-field auto|none|<name>] [--time-unit auto|s|ms|us|ns]
//
// Output per input cloud, stamped with the cloud's timestamp:
//   frame_id         odom frame (= sensor frame at the first cloud)
//   body_frame_id    the cloud's frame_id (or --body-frame)
//   pose             body in odom frame, absolute (accumulated); with deskewing
//                    it is the pose at the time of the scan's last point
//   linear/angular_velocity   per-epoch motion / dt, in the body frame
//   metadata         fit, sigma, proc_ms, dropped, scan_ms
//
// Deskew: if the cloud has a per-point time field, the times are handed to
// KISS-ICP, which moves every point to where it would have been measured at
// the end of the scan. --time-field auto takes the first field named t, time,
// timestamp, time_stamp, stamp, offset_time or time_offset (any numeric type);
// --time-field none turns deskewing off. --time-unit auto takes the largest of
// s / ms / us / ns that makes the scan last at most 1 s.
//
// The receive callback only parses the cloud into a single-slot buffer: if a
// new cloud arrives while the previous one is still being registered, the
// older unprocessed one is dropped.
// A timestamp going backwards (e.g. replay looping) resets the odometry.
//
// --viz opens the viewer (accumulated cloud + ego trajectory, see viewer.hpp
// for controls). The viewer then owns the main thread and registration moves
// to a worker thread; without --viz registration runs on the main thread.
// Closing the window only closes the viewer, the node keeps running.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <ecal/msg/protobuf/subscriber.h>

#include "foxglove/Odometry.pb.h"
#include "foxglove/PointCloud.pb.h"
#include "kiss_odom.hpp"

#ifdef PLY_ODOM_VIEWER
#include "vis_feed.hpp"
#else
class VisFeed;  // only ever passed as a null pointer in a build without the viewer
#endif

using Cloud = KissOdom::Cloud;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string in = "pointcloud";
    std::string out = "odometry";
    std::string odom_frame = "odom";
    std::string body_frame;  // empty: use the cloud's frame_id
    KissOdom::Params odom;
    double voxel = -1.0;      // default max_range / 100
    double vis_voxel = -1.0;  // default voxel / 4
    bool quiet = false;
    bool viz = false;
    std::string time_field = "auto";  // auto | none | <field name>
    std::string time_unit = "auto";   // auto | s | ms | us | ns
};

// One received cloud, parsed, waiting for registration.
struct Pending {
    Cloud cloud;
    std::vector<double> times;  // per-point time, raw field values; empty if no time field
    std::string time_field;     // name of the field `times` came from
    std::string field_names;    // all field names, for the "no time field" hint
    double stamp = -1.0;        // seconds
    google::protobuf::Timestamp ts;
    std::string frame_id;
    bool valid = false;
};

using Field = foxglove::PackedElementField;

static size_t numericSize(Field::NumericType type) {
    switch (type) {
        case Field::UINT8:
        case Field::INT8:
            return 1;
        case Field::UINT16:
        case Field::INT16:
            return 2;
        case Field::UINT32:
        case Field::INT32:
        case Field::FLOAT32:
            return 4;
        case Field::FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static double readNumeric(const char *p, Field::NumericType type) {
    switch (type) {
        case Field::UINT8: { uint8_t v; std::memcpy(&v, p, 1); return v; }
        case Field::INT8: { int8_t v; std::memcpy(&v, p, 1); return v; }
        case Field::UINT16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
        case Field::INT16: { int16_t v; std::memcpy(&v, p, 2); return v; }
        case Field::UINT32: { uint32_t v; std::memcpy(&v, p, 4); return v; }
        case Field::INT32: { int32_t v; std::memcpy(&v, p, 4); return v; }
        case Field::FLOAT32: { float v; std::memcpy(&v, p, 4); return v; }
        case Field::FLOAT64: { double v; std::memcpy(&v, p, 8); return v; }
        default: return 0.0;
    }
}

// The per-point time field: the one named `want`, or for "auto" the first of the usual names.
static const Field *findTimeField(const foxglove::PointCloud &msg, const std::string &want) {
    if (want == "none") return nullptr;
    const auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    const auto find = [&](const std::string &name) -> const Field * {
        for (const auto &field : msg.fields()) {
            if (lower(field.name()) == name) return &field;
        }
        return nullptr;
    };
    if (want != "auto") return find(lower(want));
    for (const char *name :
         {"t", "time", "timestamp", "time_stamp", "stamp", "offset_time", "time_offset"}) {
        if (const Field *f = find(name)) return f;
    }
    return nullptr;
}

// Seconds per raw unit of the time field: the largest of s / ms / us / ns for
// which a scan of `raw_range` lasts at most 1 s.
static double autoTimeUnit(double raw_range) {
    for (const double unit : {1.0, 1e-3, 1e-6}) {
        if (raw_range * unit <= 1.0) return unit;
    }
    return 1e-9;
}

// Extract x/y/z (FLOAT32 or FLOAT64) and, if present, the per-point time from
// the packed data, and move the points into frame_id if the message carries a
// non-identity pose.
static bool toCloud(const foxglove::PointCloud &msg,
                    const std::string &time_field,
                    Pending &out,
                    std::string &err) {
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

    const Field *ft = findTimeField(msg, time_field);
    if (ft && (numericSize(ft->type()) == 0 || ft->offset() + numericSize(ft->type()) > stride)) {
        ft = nullptr;  // unknown type or out of bounds: treat as absent
    }
    if (ft) {
        out.time_field = ft->name();
    } else {
        for (const auto &field : msg.fields()) out.field_names += field.name() + " ";
    }

    const std::string &data = msg.data();
    const size_t n = data.size() / stride;
    Cloud &pts = out.cloud;
    pts.clear();
    pts.reserve(n);
    out.times.clear();
    if (ft) out.times.reserve(n);
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
            pts.emplace_back(v[0], v[1], v[2]);
            if (ft) out.times.push_back(readNumeric(p + ft->offset(), ft->type()));
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
            for (auto &pt : pts) pt = T * pt;
        }
    }
    return true;
}

static int usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--in pointcloud] [--out odometry] [--odom-frame odom] [--body-frame <id>]\n"
                 "       [-r max_range_m=100] [-v voxel_m=max_range/100] [-p max_points_per_voxel=20]\n"
                 "       [-i max_icp_iterations=500] [-q (no per-frame log)]\n"
                 "       [--viz (show viewer)] [-d vis_voxel_m=voxel/4]\n"
                 "       [--time-field auto|none|<name>] [--time-unit auto|s|ms|us|ns]  (deskew)\n",
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
        } else if (is("-d") && has_val) {
            args.vis_voxel = std::atof(argv[++i]);
        } else if (is("-q")) {
            args.quiet = true;
        } else if (is("--viz")) {
            args.viz = true;
        } else if (is("--time-field") && has_val) {
            args.time_field = argv[++i];
        } else if (is("--time-unit") && has_val) {
            args.time_unit = argv[++i];
        } else {
            return usage(argv[0]);
        }
    }
    args.odom.voxel = args.voxel > 0.0 ? args.voxel : args.odom.max_range / 100.0;
    if (args.vis_voxel <= 0.0) args.vis_voxel = args.odom.voxel / 4.0;
    double time_unit = 0.0;  // seconds per raw unit of the time field, 0 = detect
    if (args.time_unit == "s") time_unit = 1.0;
    else if (args.time_unit == "ms") time_unit = 1e-3;
    else if (args.time_unit == "us") time_unit = 1e-6;
    else if (args.time_unit == "ns") time_unit = 1e-9;
    else if (args.time_unit != "auto") return usage(argv[0]);
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
        if (!toCloud(msg, args.time_field, p, err)) {
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
        // scan[ms] = scan duration used for deskewing, 0 = not deskewed
        std::printf("%7s %8s %8s %8s %6s %8s %7s %8s %7s %6s %7s %6s %7s\n", "frame", "points",
                    "step[m]", "rot[deg]", "fit[%]", "scan[ms]", "pre[ms]", "icp[ms]", "map[ms]",
                    "src", "vox", "sigma", "dropped");
    }
    std::fflush(stdout);

    // Registration loop. `feed` (may be null) gets every registered scan for the viewer.
    const auto process = [&](VisFeed *feed) {
        KissOdom odom(args.odom);
        const double rad2deg = 180.0 / 3.14159265358979323846;
        size_t n_frames = 0;
        double last_stamp = -1.0;
        bool time_unit_auto = false, deskew_reported = false;
#ifndef PLY_ODOM_VIEWER
        (void)feed;
#endif

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
#ifdef PLY_ODOM_VIEWER
                if (feed) feed->reset();
#endif
            }
            last_stamp = p.stamp;

            // Per-point times: raw field values -> seconds since the scan's first point.
            if (!p.times.empty()) {
                const auto mm = std::minmax_element(p.times.cbegin(), p.times.cend());
                const double t_min = *mm.first, range = *mm.second - *mm.first;
                if (time_unit <= 0.0 && range > 0.0) {
                    time_unit = autoTimeUnit(range);
                    time_unit_auto = true;
                }
                if (time_unit > 0.0 && range > 0.0) {
                    for (double &t : p.times) t = (t - t_min) * time_unit;
                } else {
                    p.times.clear();  // constant field: nothing to deskew with
                }
                if (!deskew_reported && !p.times.empty()) {
                    const char *unit = time_unit == 1.0    ? "s"
                                       : time_unit == 1e-3 ? "ms"
                                       : time_unit == 1e-6 ? "us"
                                                           : "ns";
                    std::printf("deskew on: time field '%s', unit %s%s, scan duration %.1f ms\n",
                                p.time_field.c_str(), unit, time_unit_auto ? " (auto)" : "",
                                1e3 * range * time_unit);
                    std::fflush(stdout);
                    deskew_reported = true;
                }
            } else if (!deskew_reported && args.time_field != "none") {
                std::printf("deskew off: no per-point time field%s%s in the cloud (fields: %s)\n",
                            args.time_field == "auto" ? "" : " ",
                            args.time_field == "auto" ? "" : args.time_field.c_str(),
                            p.field_names.c_str());
                std::fflush(stdout);
                deskew_reported = true;
            }

            const auto t0 = Clock::now();
            const KissOdom::Result r = odom.registerFrame(p.cloud, p.stamp, p.times);
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
            meta("scan_ms", std::to_string(1e3 * r.scan_s));
            pub.Send(msg);

            // Process state for the eCAL monitor: warn when the registration looks poor.
            char info[128];
            std::snprintf(info, sizeof(info), "frame %zu, fit %.0f %%, %.0f ms, dropped %zu", n_frames,
                          100.0 * r.fit, proc_ms, n_dropped);
            eCAL::Process::SetState(r.first || r.fit >= 0.5 ? proc_sev_healthy : proc_sev_warning,
                                    proc_sev_level1, info);
#ifdef PLY_ODOM_VIEWER
            if (feed) feed->push(r.frame, r.pose, info);
#endif

            if (!args.quiet) {
                std::printf("%7zu %8zu ", n_frames, p.cloud.size());
                if (r.first) {
                    std::printf("%8s %8s %6s ", "-", "-", "-");
                } else {
                    std::printf("%8.3f %8.3f %6.1f ", r.delta.translation().norm(),
                                r.delta.so3().log().norm() * rad2deg, 100.0 * r.fit);
                }
                std::printf("%8.1f %7.1f %8.1f %7.1f %6zu %7zu %6.2f %7zu\n", 1e3 * r.scan_s, r.ms_pre,
                            r.ms_icp, r.ms_map, r.n_src, r.n_voxels, r.sigma, n_dropped);
                std::fflush(stdout);
            }
        }
    };

    if (args.viz) {
#ifdef PLY_ODOM_VIEWER
        // Keep Ctrl-C a plain SIGINT instead of an SDL "window closed" event.
        SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
        Viewer viewer;
        if (viewer.init("kiss_odom_node")) {
            VisFeed feed(args.vis_voxel);
            viewer.setStatus("waiting for point clouds");
            std::thread worker([&] { process(&feed); });
            while (eCAL::Ok() && viewer.frame()) feed.drain(viewer);
            feed.close();  // window closed: the node keeps running without the viewer
            viewer.shutdown();
            worker.join();
            eCAL::Finalize();
            return 0;
        }
        std::fprintf(stderr, "no viewer, running without\n");
#else
        std::fprintf(stderr, "--viz ignored: built without the viewer (SDL2 / OpenGL not found)\n");
#endif
    }
    process(nullptr);

    eCAL::Finalize();
    return 0;
}
