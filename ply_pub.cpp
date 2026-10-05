// ply_pub - replays a folder of PLY scans as foxglove.PointCloud on an eCAL 5
// topic. Test input for kiss_odom_node.
//
//   ply_pub <dir | a.ply b.ply ...> [--out pointcloud] [--hz 10] [--rate <hz>] [--frame lidar] [--loop]
//
// Points go out as x/y/z FLOAT32, stamped start + i / hz. --rate sets how fast
// they are actually sent (default: hz), e.g. to replay slower than real time.
// If the PLY vertices have a float/double property t, time or timestamp, it
// is sent along as a FLOAT64 field "t" (per-point time, for deskewing).
// With --loop the stamps restart on every pass, like a looping recording.
// Waits up to 5 s for a subscriber before the first cloud.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>

#include "foxglove/PointCloud.pb.h"
#include "ply_io.hpp"

int main(int argc, char **argv) {
    std::vector<std::string> files;
    std::string topic = "pointcloud", frame_id = "lidar";
    double hz = 10.0, rate = -1.0;
    bool loop = false;
    for (int i = 1; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        if (!std::strcmp(argv[i], "--out") && has_val) {
            topic = argv[++i];
        } else if (!std::strcmp(argv[i], "--hz") && has_val) {
            hz = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rate") && has_val) {
            rate = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--frame") && has_val) {
            frame_id = argv[++i];
        } else if (!std::strcmp(argv[i], "--loop")) {
            loop = true;
        } else if (argv[i][0] != '-' && std::filesystem::is_directory(argv[i])) {
            const auto found = listPly(argv[i]);
            files.insert(files.end(), found.begin(), found.end());
        } else if (argv[i][0] != '-') {
            files.emplace_back(argv[i]);
        } else {
            files.clear();
            break;
        }
    }
    if (rate <= 0.0) rate = hz;
    if (files.empty() || hz <= 0.0) {
        std::fprintf(stderr,
                     "usage: %s <dir | a.ply b.ply ...> [--out pointcloud] [--hz 10] [--rate <hz>] "
                     "[--frame lidar] [--loop]\n",
                     argv[0]);
        return 1;
    }

    eCAL::Initialize(0, nullptr, "ply_pub");
    eCAL::protobuf::CPublisher<foxglove::PointCloud> pub(topic);
    for (int i = 0; i < 50 && eCAL::Ok() && !pub.IsSubscribed(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / rate));
    const double t_start = std::chrono::duration<double>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    auto next = Clock::now();
    size_t n_sent = 0;

    do {
        for (size_t i = 0; i < files.size() && eCAL::Ok(); ++i) {
            std::vector<Eigen::Vector3d> pts;
            std::vector<double> times;
            try {
                pts = loadPly(files[i], &times);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "skip %s: %s\n", files[i].c_str(), e.what());
                continue;
            }

            foxglove::PointCloud msg;
            const double stamp = t_start + static_cast<double>(i) / hz;  // restarts on --loop
            msg.mutable_timestamp()->set_seconds(static_cast<int64_t>(stamp));
            msg.mutable_timestamp()->set_nanos(
                static_cast<int32_t>((stamp - static_cast<double>(static_cast<int64_t>(stamp))) * 1e9));
            msg.set_frame_id(frame_id);
            msg.mutable_pose()->mutable_orientation()->set_w(1.0);
            const bool has_t = times.size() == pts.size() && !pts.empty();
            const uint32_t stride = has_t ? 20 : 12;
            msg.set_point_stride(stride);
            const char *names[3] = {"x", "y", "z"};
            for (uint32_t k = 0; k < 3; ++k) {
                auto *f = msg.add_fields();
                f->set_name(names[k]);
                f->set_offset(4 * k);
                f->set_type(foxglove::PackedElementField::FLOAT32);
            }
            if (has_t) {
                auto *f = msg.add_fields();
                f->set_name("t");
                f->set_offset(12);
                f->set_type(foxglove::PackedElementField::FLOAT64);
            }
            std::string *data = msg.mutable_data();
            data->resize(pts.size() * stride);
            char *dst = &(*data)[0];
            for (size_t k = 0; k < pts.size(); ++k, dst += stride) {
                const float xyz[3] = {static_cast<float>(pts[k].x()), static_cast<float>(pts[k].y()),
                                      static_cast<float>(pts[k].z())};
                std::memcpy(dst, xyz, 12);
                if (has_t) std::memcpy(dst + 12, &times[k], 8);
            }

            std::this_thread::sleep_until(next);
            next += period;
            pub.Send(msg);
            ++n_sent;
            std::printf("\rsent %zu  %s (%zu pts)   ", n_sent,
                        std::filesystem::path(files[i]).filename().string().c_str(), pts.size());
            std::fflush(stdout);
        }
    } while (loop && eCAL::Ok());

    std::printf("\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // let the last one out
    eCAL::Finalize();
    return 0;
}
