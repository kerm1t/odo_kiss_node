// viewer.hpp - minimal SDL2 + OpenGL 3.3 core viewer: accumulated point cloud,
// ego trajectory and current pose axes, a 10 m ground grid, and a text overlay
// with driven distance, velocity and acceleration. Header-only, no GL loader
// needed on Linux / macOS.
//
//   left drag   orbit            F    follow ego on/off
//   right drag  pan              R    reset view
//   wheel       zoom             +/-  point size
//   Esc / Q     quit             G    grid on/off
//
// World is z-up. Points are (x, y, z, s) with s = scalar used for colouring
// (height). If poses come with timestamps, the trajectory is coloured by
// velocity (legend bottom left); without, it is white and velocity and
// acceleration show "-".
//
// Overlay numbers: distance = path length. Velocity = distance over the last
// 0.5 s (at least one frame step). Acceleration = change of that velocity
// against the 0.5 s before, i.e. along the path and smoothed over ~1 s; the
// raw per-frame difference would mostly be registration noise.
//
// Cost control: the cloud only grows, so the viewer redraws on demand instead
// of every display frame - immediately on input, and for new data at most
// ~20 % of the time (measured draw cost). While dragging, a subsampled cloud
// is drawn. Points live in the GPU buffer only (no CPU copy).
#pragma once

#define SDL_MAIN_HANDLED
#include <SDL.h>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "viewer_font.hpp"

class Viewer {
public:
    Viewer() = default;
    Viewer(const Viewer &) = delete;
    Viewer &operator=(const Viewer &) = delete;
    ~Viewer() { shutdown(); }

    bool init(const char *title, int w = 1280, int h = 800) {
        SDL_SetMainReady();
        if (SDL_Init(SDL_INIT_VIDEO) != 0) return fail("SDL_Init");
        sdl_up_ = true;
        // Without a display SDL silently picks its offscreen driver: nothing to look at.
        const char *drv = SDL_GetCurrentVideoDriver();
        if (!drv || !std::strcmp(drv, "offscreen") || !std::strcmp(drv, "dummy")) {
            std::fprintf(stderr, "viewer: no display\n");
            shutdown();
            return false;
        }
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        win_ = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h,
                                SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                                    SDL_WINDOW_ALLOW_HIGHDPI);
        if (!win_) return fail("SDL_CreateWindow");
        ctx_ = SDL_GL_CreateContext(win_);
        if (!ctx_) return fail("SDL_GL_CreateContext");
        SDL_GL_SetSwapInterval(1);
        title_ = title;

        prog_ = buildProgram();
        if (!prog_) return false;
        u_mvp_ = glGetUniformLocation(prog_, "mvp");
        u_range_ = glGetUniformLocation(prog_, "range");
        u_flat_ = glGetUniformLocation(prog_, "flat_col");
        u_psize_ = glGetUniformLocation(prog_, "psize");
        u_cmap_ = glGetUniformLocation(prog_, "cmap");

        makeBuffer(vao_pts_, vbo_pts_);
        makeBuffer(vao_traj_, vbo_traj_);
        makeBuffer(vao_axes_, vbo_axes_);
        makeBuffer(vao_aux_, vbo_aux_);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        resetView();
        return true;
    }

    // Append points, 4 floats each: world x, y, z + colour scalar.
    void addPoints(const std::vector<float> &xyzs) {
        if (xyzs.empty()) return;
        if (!range_set_) setRangeFrom(xyzs);
        const size_t old_bytes = n_pts_ * kStride;
        const size_t add_bytes = xyzs.size() * sizeof(float);
        if (old_bytes + add_bytes > cap_bytes_) {  // grow x2, copy GPU -> GPU
            const size_t cap = std::max({old_bytes + add_bytes, 2 * cap_bytes_, size_t{16} << 20});
            GLuint nb = 0;
            glGenBuffers(1, &nb);
            glBindBuffer(GL_COPY_WRITE_BUFFER, nb);
            glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(cap), nullptr,
                         GL_DYNAMIC_DRAW);
            if (old_bytes > 0) {
                glBindBuffer(GL_COPY_READ_BUFFER, vbo_pts_);
                glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                                    static_cast<GLsizeiptr>(old_bytes));
            }
            glDeleteBuffers(1, &vbo_pts_);
            vbo_pts_ = nb;  // attribute pointer is re-set at draw time
            cap_bytes_ = cap;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts_);
        glBufferSubData(GL_ARRAY_BUFFER, static_cast<GLintptr>(old_bytes),
                        static_cast<GLsizeiptr>(add_bytes), xyzs.data());
        n_pts_ += xyzs.size() / 4;
        data_dirty_ = true;
    }

    // Append an ego pose (sensor in world). Extends the trajectory.
    // `stamp` in seconds, < 0 if unknown (then no velocity for this step).
    void addPose(const Eigen::Matrix4f &T, double stamp = -1.0) {
        float speed = 0.0f;
        if (has_pose_) {
            const double step = (T.block<3, 1>(0, 3) - pose_.block<3, 1>(0, 3)).norm();
            path_m_ += step;
            if (stamp >= 0.0 && last_stamp_ >= 0.0 && stamp > last_stamp_) {
                speed = static_cast<float>(step / (stamp - last_stamp_));
                if (traj_.size() == 4) traj_[3] = speed;  // first node: speed of the first step
                has_speed_ = true;
                v_max_ = std::max(v_max_, speed);
            }
        }
        if (stamp >= 0.0) {
            hist_.emplace_back(stamp, path_m_);
            while (hist_.size() > 4 && hist_.front().first < stamp - 5.0) hist_.pop_front();
            last_stamp_ = stamp;
        }
        pose_ = T;
        has_pose_ = true;
        traj_.insert(traj_.end(), {T(0, 3), T(1, 3), T(2, 3), speed});
        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(traj_.size() * sizeof(float)),
                     traj_.data(), GL_DYNAMIC_DRAW);
        data_dirty_ = true;
    }

    // Drop all points and the trajectory (odometry started over). Keeps the camera
    // and the colour range.
    void clear() {
        n_pts_ = 0;
        traj_.clear();
        has_pose_ = false;
        pose_.setIdentity();
        path_m_ = 0.0;
        hist_.clear();
        last_stamp_ = -1.0;
        has_speed_ = false;
        v_max_ = 0.0f;
        data_dirty_ = true;
    }

    void setStatus(const std::string &s) {
        status_ = s;
        title_dirty_ = true;
    }

    size_t numPoints() const { return n_pts_; }

    // Handle input and redraw if needed, otherwise sleep a few ms.
    // Returns false once the user wants to quit.
    bool frame() {
        if (!handleEvents()) return false;
        if (title_dirty_) {
            SDL_SetWindowTitle(win_, (title_ + "  |  " + status_ + (follow_ ? "  |  follow" : ""))
                                         .c_str());
            title_dirty_ = false;
        }

        const Uint32 t_start = SDL_GetTicks();
        const bool data_due = data_dirty_ && SDL_TICKS_PASSED(t_start, next_data_draw_);
        if (!view_dirty_ && !data_due) {
            SDL_Delay(5);
            return true;
        }
        if (follow_ && has_pose_) target_ = pose_.block<3, 1>(0, 3);

        int w = 1, h = 1;
        SDL_GL_GetDrawableSize(win_, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const Eigen::Vector3f eye = target_ + dist_ * Eigen::Vector3f(std::cos(pitch_) * std::cos(yaw_),
                                                                      std::cos(pitch_) * std::sin(yaw_),
                                                                      std::sin(pitch_));
        const float znear = std::max(0.05f, dist_ * 0.01f);
        const Eigen::Matrix4f mvp =
            perspective(0.87f, static_cast<float>(w) / static_cast<float>(std::max(h, 1)), znear,
                        znear * 1.0e5f) *
            lookAt(eye, target_, Eigen::Vector3f::UnitZ());

        glUseProgram(prog_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
        glUniform2f(u_range_, range_lo_, range_hi_);

        // accumulated cloud, coloured by scalar
        glEnable(GL_DEPTH_TEST);
        glUniform4f(u_flat_, 0, 0, 0, 0);
        glUniform1f(u_psize_, psize_);
        // While dragging draw every k-th point only (points arrive unordered
        // within a scan, so a stride thins the cloud evenly).
        const size_t k =
            dragging_ ? std::max<size_t>((n_pts_ + kDragBudget - 1) / kDragBudget, 1) : 1;
        glBindVertexArray(vao_pts_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts_);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(k * kStride),
                              nullptr);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(n_pts_ / k));

        // grid, trajectory and ego axes go on top of the cloud
        glDisable(GL_DEPTH_TEST);
        if (show_grid_) drawGrid();
        const GLsizei n_traj = static_cast<GLsizei>(traj_.size() / 4);
        const float v_hi = speedScaleMax();
        glBindVertexArray(vao_traj_);
        if (has_speed_) {
            // coloured by velocity, with a dark rim so the colours read on any background
            glUniform4f(u_flat_, 0.0f, 0.0f, 0.0f, 1.0f);
            glUniform1f(u_psize_, 8.0f);
            glDrawArrays(GL_POINTS, 0, n_traj);
            glUniform4f(u_flat_, 0, 0, 0, 0);
            glUniform1i(u_cmap_, 1);
            glUniform2f(u_range_, 0.0f, v_hi);
            glDrawArrays(GL_LINE_STRIP, 0, n_traj);
            glUniform1f(u_psize_, 5.0f);
            glDrawArrays(GL_POINTS, 0, n_traj);
            glUniform1i(u_cmap_, 0);
        } else {
            glUniform4f(u_flat_, 1.0f, 1.0f, 1.0f, 1.0f);
            glDrawArrays(GL_LINE_STRIP, 0, n_traj);
            glUniform1f(u_psize_, 4.0f);
            glDrawArrays(GL_POINTS, 0, n_traj);
        }
        if (n_traj > 0) {  // current position
            glUniform4f(u_flat_, 1.0f, 0.85f, 0.1f, 1.0f);
            glUniform1f(u_psize_, 10.0f);
            glDrawArrays(GL_POINTS, n_traj - 1, 1);
        }

        if (has_pose_) {
            const float len = std::max(1.0f, dist_ * 0.06f);
            const Eigen::Vector3f o = pose_.block<3, 1>(0, 3);
            float v[24];
            for (int a = 0; a < 3; ++a) {
                const Eigen::Vector3f e = o + len * pose_.block<3, 1>(0, a);
                const float seg[8] = {o.x(), o.y(), o.z(), 0, e.x(), e.y(), e.z(), 0};
                std::copy(seg, seg + 8, v + 8 * a);
            }
            glBindVertexArray(vao_axes_);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_axes_);
            glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_DYNAMIC_DRAW);
            const float rgb[3][3] = {{1.0f, 0.25f, 0.25f}, {0.3f, 1.0f, 0.3f}, {0.35f, 0.55f, 1.0f}};
            for (int a = 0; a < 3; ++a) {
                glUniform4f(u_flat_, rgb[a][0], rgb[a][1], rgb[a][2], 1.0f);
                glDrawArrays(GL_LINES, 2 * a, 2);
            }
        }

        drawOverlay(w, h, v_hi);
        glBindVertexArray(0);

        // Measure what this draw really cost and hold back data-driven redraws
        // so they take at most ~20 % of the time; input redraws are not held back.
        glFinish();
        const Uint32 cost = SDL_GetTicks() - t_start;
        SDL_GL_SwapWindow(win_);
        view_dirty_ = data_dirty_ = false;
        next_data_draw_ = SDL_GetTicks() + 4 * cost;
        return true;
    }

    void shutdown() {
        if (ctx_) {
            SDL_GL_DeleteContext(ctx_);
            ctx_ = nullptr;
        }
        if (win_) {
            SDL_DestroyWindow(win_);
            win_ = nullptr;
        }
        if (sdl_up_) {
            SDL_Quit();
            sdl_up_ = false;
        }
    }

private:
    bool fail(const char *what) {
        std::fprintf(stderr, "viewer: %s failed: %s\n", what, SDL_GetError());
        shutdown();
        return false;
    }

    void resetView() {
        yaw_ = 3.14159265f;  // camera behind the sensor, looking along +x
        pitch_ = 0.6f;
        dist_ = 80.0f;
        follow_ = true;
        title_dirty_ = true;
        view_dirty_ = true;
    }

    bool handleEvents() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    return false;
                case SDL_WINDOWEVENT:  // exposed, resized, ...
                    view_dirty_ = true;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                    dragging_ = SDL_GetMouseState(nullptr, nullptr) != 0;
                    view_dirty_ = true;  // full-detail redraw on release
                    break;
                case SDL_KEYDOWN:
                    view_dirty_ = true;
                    switch (e.key.keysym.sym) {
                        case SDLK_ESCAPE:
                        case SDLK_q:
                            return false;
                        case SDLK_f:
                            follow_ = !follow_;
                            title_dirty_ = true;
                            break;
                        case SDLK_r:
                            resetView();
                            break;
                        case SDLK_g:
                            show_grid_ = !show_grid_;
                            break;
                        case SDLK_PLUS:
                        case SDLK_EQUALS:
                        case SDLK_KP_PLUS:
                            psize_ = std::min(psize_ + 1.0f, 8.0f);
                            break;
                        case SDLK_MINUS:
                        case SDLK_KP_MINUS:
                            psize_ = std::max(psize_ - 1.0f, 1.0f);
                            break;
                        default:
                            break;
                    }
                    break;
                case SDL_MOUSEWHEEL:
                    dist_ = std::clamp(dist_ * std::exp(-0.12f * static_cast<float>(e.wheel.y)),
                                       0.5f, 5000.0f);
                    view_dirty_ = true;
                    break;
                case SDL_MOUSEMOTION: {
                    const float dx = static_cast<float>(e.motion.xrel);
                    const float dy = static_cast<float>(e.motion.yrel);
                    if (e.motion.state & SDL_BUTTON_LMASK) {
                        yaw_ -= 0.005f * dx;
                        pitch_ = std::clamp(pitch_ + 0.005f * dy, -1.55f, 1.55f);
                        view_dirty_ = true;
                    } else if (e.motion.state & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) {
                        // pan in the ground plane, like dragging a map
                        const Eigen::Vector3f fwd(-std::cos(yaw_), -std::sin(yaw_), 0.0f);
                        const Eigen::Vector3f right(fwd.y(), -fwd.x(), 0.0f);
                        target_ += (fwd * dy - right * dx) * dist_ * 0.0015f;
                        if (follow_) title_dirty_ = true;
                        follow_ = false;
                        view_dirty_ = true;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return true;
    }

    // ---- ground grid ----------------------------------------------------

    // 10 m grid (every 100 m a stronger line) around the camera target, at the
    // estimated ground height (low end of the height colour range), in the
    // odometry frame's x/y plane. Past ~600 m camera distance only the 100 m lines.
    // Drawn see-through over the cloud, so the scale stays readable on the ground points.
    void drawGrid() {
        const float z = range_set_ ? range_lo_ : 0.0f;
        const float half = std::clamp(3.0f * dist_, 100.0f, 1500.0f);
        const int x0 = static_cast<int>(std::floor((target_.x() - half) / 10.0f));
        const int x1 = static_cast<int>(std::ceil((target_.x() + half) / 10.0f));
        const int y0 = static_cast<int>(std::floor((target_.y() - half) / 10.0f));
        const int y1 = static_cast<int>(std::ceil((target_.y() + half) / 10.0f));
        const bool minor = dist_ < 600.0f;
        std::vector<float> lines[2];  // 0: 10 m, 1: 100 m
        const auto add = [&](std::vector<float> &o, float ax, float ay, float bx, float by) {
            o.insert(o.end(), {ax, ay, z, 0.0f, bx, by, z, 0.0f});
        };
        const float fx0 = 10.0f * static_cast<float>(x0), fx1 = 10.0f * static_cast<float>(x1);
        const float fy0 = 10.0f * static_cast<float>(y0), fy1 = 10.0f * static_cast<float>(y1);
        for (int i = x0; i <= x1; ++i) {
            const bool major = i % 10 == 0;
            const float x = 10.0f * static_cast<float>(i);
            if (major || minor) add(lines[major], x, fy0, x, fy1);
        }
        for (int i = y0; i <= y1; ++i) {
            const bool major = i % 10 == 0;
            const float y = 10.0f * static_cast<float>(i);
            if (major || minor) add(lines[major], fx0, y, fx1, y);
        }
        const GLsizei n0 = static_cast<GLsizei>(lines[0].size() / 4);
        const GLsizei n1 = static_cast<GLsizei>(lines[1].size() / 4);
        lines[0].insert(lines[0].end(), lines[1].begin(), lines[1].end());
        glBindVertexArray(vao_aux_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_aux_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(lines[0].size() * sizeof(float)),
                     lines[0].data(), GL_DYNAMIC_DRAW);
        glEnable(GL_BLEND);
        glUniform4f(u_flat_, 0.75f, 0.78f, 0.90f, 0.22f);
        glDrawArrays(GL_LINES, 0, n0);
        glUniform4f(u_flat_, 0.75f, 0.78f, 0.90f, 0.50f);
        glDrawArrays(GL_LINES, n0, n1);
        glDisable(GL_BLEND);
    }

    // ---- kinematics for the overlay ---------------------------------------

    // Upper end of the velocity colour scale: fastest step so far, rounded up to 5 m/s.
    float speedScaleMax() const { return std::max(5.0f, 5.0f * std::ceil(v_max_ / 5.0f)); }

    // Path length at time t, interpolated in the recent history.
    bool pathAt(double t, double &d) const {
        if (hist_.empty() || t < hist_.front().first) return false;
        for (size_t i = hist_.size(); i-- > 1;) {
            if (hist_[i - 1].first <= t) {
                const double span = hist_[i].first - hist_[i - 1].first;
                const double u = span > 0.0 ? std::min((t - hist_[i - 1].first) / span, 1.0) : 0.0;
                d = hist_[i - 1].second + u * (hist_[i].second - hist_[i - 1].second);
                return true;
            }
        }
        d = hist_.front().second;
        return true;
    }

    // Velocity over the last 0.5 s (at least one frame step) and its change
    // against the window before.
    void kinematics(bool &has_v, double &vel, bool &has_a, double &acc) const {
        has_v = has_a = false;
        if (hist_.size() < 2) return;
        const double t1 = hist_.back().first, d1 = hist_.back().second;
        const double w = std::max(0.5, t1 - hist_[hist_.size() - 2].first);
        double d0 = 0.0, dm = 0.0;
        if (!pathAt(t1 - w, d0)) {  // not enough history yet: use what there is
            if (t1 > hist_.front().first) {
                vel = (d1 - hist_.front().second) / (t1 - hist_.front().first);
                has_v = true;
            }
            return;
        }
        vel = (d1 - d0) / w;
        has_v = true;
        if (pathAt(t1 - 2.0 * w, dm)) {
            acc = ((d1 - d0) - (d0 - dm)) / (w * w);
            has_a = true;
        }
    }

    // ---- 2D overlay: numbers top left, velocity legend bottom left --------

    // One GL point per lit font pixel; x, y = top left of the text in pixels.
    static void addText(std::vector<float> &pts, float x, float y, float s, const std::string &text) {
        for (const char ch : text) {
            if (ch >= 32 && ch <= 126) {
                const unsigned short *rows = kFont[ch - 32];
                for (int r = 0; r < kFontH; ++r) {
                    for (int c = 0; c < kFontW; ++c) {
                        if (rows[r] >> (kFontW - 1 - c) & 1) {
                            pts.insert(pts.end(), {x + (static_cast<float>(c) + 0.5f) * s,
                                                   y + (static_cast<float>(r) + 0.5f) * s, 0.0f, 0.0f});
                        }
                    }
                }
            }
            x += static_cast<float>(kFontW) * s;
        }
    }

    // Two triangles; `s0` / `s1` = colour scalar at the bottom / top edge.
    static void addRect(std::vector<float> &tris, float x0, float y0, float x1, float y1,
                        float s0 = 0.0f, float s1 = 0.0f) {
        tris.insert(tris.end(), {x0, y0, 0, s1, x1, y0, 0, s1, x1, y1, 0, s0,
                                 x0, y0, 0, s1, x1, y1, 0, s0, x0, y1, 0, s0});
    }

    void drawOverlay(int w, int h, float v_hi) {
        int win_w = w, win_h = h;
        SDL_GetWindowSize(win_, &win_w, &win_h);
        // UI scale: 2 on HiDPI displays, where the drawable is larger than the window
        const float s = std::max(
            1.0f, std::round(static_cast<float>(w) / static_cast<float>(std::max(win_w, 1))));
        const float cw = static_cast<float>(kFontW) * s, lh = static_cast<float>(kFontH + 3) * s;
        const float m = 10.0f * s;
        std::vector<float> panel, bar, marks, text;

        // numbers
        bool has_v = false, has_a = false;
        double vel = 0.0, acc = 0.0;
        kinematics(has_v, vel, has_a, acc);
        char line[3][64];
        std::snprintf(line[0], sizeof(line[0]), "dist %9.1f m", path_m_);
        if (has_v) {
            std::snprintf(line[1], sizeof(line[1]), "vel  %9.2f m/s %7.1f km/h", vel, 3.6 * vel);
        } else {
            std::snprintf(line[1], sizeof(line[1]), "vel  %9s m/s %7s km/h", "-", "-");
        }
        if (has_a) {
            std::snprintf(line[2], sizeof(line[2]), "acc  %+9.2f m/s^2", acc);
        } else {
            std::snprintf(line[2], sizeof(line[2]), "acc  %9s m/s^2", "-");
        }
        addRect(panel, m, m, m + 34.0f * cw, m + 3.0f * lh + 9.0f * s);
        for (int i = 0; i < 3; ++i) {
            addText(text, m + cw, m + 6.0f * s + static_cast<float>(i) * lh, s, line[i]);
        }

        // velocity legend: colour bar with 0, half and full scale in m/s and km/h
        if (has_speed_) {
            const float bw = 14.0f * s, bh = 150.0f * s;
            const float bx = m + cw, by1 = static_cast<float>(h) - m - 8.0f * s, by0 = by1 - bh;
            addRect(panel, m, by0 - lh - 12.0f * s, m + 27.0f * cw, static_cast<float>(h) - m);
            addText(text, bx, by0 - lh - 4.0f * s, s, "velocity");
            const int n = 32;
            for (int i = 0; i < n; ++i) {
                const float f0 = static_cast<float>(i) / static_cast<float>(n);
                const float f1 = static_cast<float>(i + 1) / static_cast<float>(n);
                addRect(bar, bx, by1 - f1 * bh, bx + bw, by1 - f0 * bh, f0, f1);
            }
            for (int i = 0; i <= 2; ++i) {
                const float f = 0.5f * static_cast<float>(i);
                const float y = by1 - f * bh;
                addRect(marks, bx + bw, y - 0.5f * s, bx + bw + 5.0f * s, y + 0.5f * s);
                char label[48];
                std::snprintf(label, sizeof(label), "%4.4g m/s %5.4g km/h",
                              static_cast<double>(f * v_hi), 3.6 * static_cast<double>(f * v_hi));
                addText(text, bx + bw + 9.0f * s, y - 0.5f * static_cast<float>(kFontH) * s, s, label);
            }
        }

        // pixel coordinates, origin top left
        Eigen::Matrix4f ortho = Eigen::Matrix4f::Identity();
        ortho(0, 0) = 2.0f / static_cast<float>(w);
        ortho(0, 3) = -1.0f;
        ortho(1, 1) = -2.0f / static_cast<float>(h);
        ortho(1, 3) = 1.0f;
        ortho(2, 2) = 0.0f;
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, ortho.data());

        const GLsizei n_panel = static_cast<GLsizei>(panel.size() / 4);
        const GLsizei n_bar = static_cast<GLsizei>(bar.size() / 4);
        const GLsizei n_marks = static_cast<GLsizei>(marks.size() / 4);
        const GLsizei n_text = static_cast<GLsizei>(text.size() / 4);
        panel.insert(panel.end(), bar.begin(), bar.end());
        panel.insert(panel.end(), marks.begin(), marks.end());
        panel.insert(panel.end(), text.begin(), text.end());
        glBindVertexArray(vao_aux_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_aux_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(panel.size() * sizeof(float)),
                     panel.data(), GL_DYNAMIC_DRAW);

        glEnable(GL_BLEND);  // see-through backdrop
        glUniform4f(u_flat_, 0.04f, 0.04f, 0.06f, 0.72f);
        glDrawArrays(GL_TRIANGLES, 0, n_panel);
        glDisable(GL_BLEND);
        glUniform4f(u_flat_, 0, 0, 0, 0);
        glUniform1i(u_cmap_, 1);
        glUniform2f(u_range_, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, n_panel, n_bar);
        glUniform1i(u_cmap_, 0);
        glUniform4f(u_flat_, 0.92f, 0.93f, 0.95f, 1.0f);
        glDrawArrays(GL_TRIANGLES, n_panel + n_bar, n_marks);
        glUniform1f(u_psize_, s);
        glDrawArrays(GL_POINTS, n_panel + n_bar + n_marks, n_text);
    }

    // Colour range = 2nd..98th percentile of the scalar in the first batch.
    void setRangeFrom(const std::vector<float> &xyzs) {
        std::vector<float> s;
        s.reserve(xyzs.size() / 4);
        for (size_t i = 3; i < xyzs.size(); i += 4) s.push_back(xyzs[i]);
        const auto pct = [&s](double p) {
            const auto it = s.begin() + static_cast<std::ptrdiff_t>(p * static_cast<double>(s.size() - 1));
            std::nth_element(s.begin(), it, s.end());
            return *it;
        };
        range_lo_ = pct(0.02);
        range_hi_ = std::max(pct(0.98), range_lo_ + 0.1f);
        range_set_ = true;
    }

    static void makeBuffer(GLuint &vao, GLuint &vbo) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glBindVertexArray(0);
    }

    static GLuint compile(GLenum type, const char *src) {
        const GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            std::fprintf(stderr, "viewer: shader compile failed:\n%s\n", log);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    static GLuint buildProgram() {
        static const char *vs = R"(#version 330 core
layout(location = 0) in vec4 a;   // xyz world, w = colour scalar
uniform mat4 mvp;
uniform vec2 range;               // scalar lo, hi
uniform vec4 flat_col;            // alpha > 0: use this colour instead
uniform float psize;
uniform int cmap;                 // 0: height (jet), 1: velocity
out vec4 col;
void main() {
    gl_Position = mvp * vec4(a.xyz, 1.0);
    gl_PointSize = psize;
    float t = clamp((a.w - range.x) / (range.y - range.x), 0.0, 1.0);
    vec3 c;
    if (cmap == 1) {  // violet -> pink -> yellow, to stand apart from the height colours
        const vec3 c0 = vec3(0.55, 0.30, 1.00), c1 = vec3(1.00, 0.30, 0.55), c2 = vec3(1.00, 0.95, 0.30);
        c = t < 0.5 ? mix(c0, c1, 2.0 * t) : mix(c1, c2, 2.0 * t - 1.0);
    } else {
        float u = mix(0.12, 0.88, t);
        c = clamp(1.5 - abs(4.0 * u - vec3(3.0, 2.0, 1.0)), 0.0, 1.0);
    }
    col = flat_col.a > 0.0 ? flat_col : vec4(c, 1.0);
})";
        static const char *fs = R"(#version 330 core
in vec4 col;
out vec4 frag;
void main() { frag = col; })";
        const GLuint v = compile(GL_VERTEX_SHADER, vs);
        const GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) return 0;
        const GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            std::fprintf(stderr, "viewer: shader link failed\n");
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    static Eigen::Matrix4f lookAt(const Eigen::Vector3f &eye,
                                  const Eigen::Vector3f &center,
                                  const Eigen::Vector3f &up) {
        const Eigen::Vector3f f = (center - eye).normalized();
        const Eigen::Vector3f s = f.cross(up).normalized();
        const Eigen::Vector3f u = s.cross(f);
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m.block<1, 3>(0, 0) = s.transpose();
        m.block<1, 3>(1, 0) = u.transpose();
        m.block<1, 3>(2, 0) = -f.transpose();
        m(0, 3) = -s.dot(eye);
        m(1, 3) = -u.dot(eye);
        m(2, 3) = f.dot(eye);
        return m;
    }

    static Eigen::Matrix4f perspective(float fovy, float aspect, float n, float f) {
        const float t = 1.0f / std::tan(0.5f * fovy);
        Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
        m(0, 0) = t / aspect;
        m(1, 1) = t;
        m(2, 2) = (f + n) / (n - f);
        m(2, 3) = 2.0f * f * n / (n - f);
        m(3, 2) = -1.0f;
        return m;
    }

    SDL_Window *win_ = nullptr;
    SDL_GLContext ctx_ = nullptr;
    bool sdl_up_ = false;
    std::string title_, status_;
    bool title_dirty_ = true;

    // redraw control
    bool view_dirty_ = true;     // camera / window changed: redraw now
    bool data_dirty_ = false;    // new points or poses: redraw when due
    bool dragging_ = false;      // mouse button held: draw subsampled
    Uint32 next_data_draw_ = 0;  // SDL ticks

    GLuint prog_ = 0;
    GLint u_mvp_ = -1, u_range_ = -1, u_flat_ = -1, u_psize_ = -1, u_cmap_ = -1;
    GLuint vao_pts_ = 0, vbo_pts_ = 0, vao_traj_ = 0, vbo_traj_ = 0, vao_axes_ = 0, vbo_axes_ = 0;
    GLuint vao_aux_ = 0, vbo_aux_ = 0;  // grid lines, then the 2D overlay

    static constexpr size_t kStride = 4 * sizeof(float);  // bytes per point
    static constexpr size_t kDragBudget = 2000000;        // points drawn while dragging
    size_t n_pts_ = 0;      // points in vbo_pts_ (GPU only)
    size_t cap_bytes_ = 0;  // allocated size of vbo_pts_
    std::vector<float> traj_;  // x y z speed
    double path_m_ = 0.0;      // driven distance
    double last_stamp_ = -1.0;
    std::deque<std::pair<double, double>> hist_;  // (stamp, path length), last few seconds
    bool has_speed_ = false;   // poses came with timestamps
    float v_max_ = 0.0f;       // fastest step so far [m/s]
    bool show_grid_ = true;
    Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();
    bool has_pose_ = false;

    float range_lo_ = 0.0f, range_hi_ = 1.0f;
    bool range_set_ = false;
    float psize_ = 2.0f;

    // orbit camera
    Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
    float yaw_ = 0.0f, pitch_ = 0.0f, dist_ = 1.0f;
    bool follow_ = true;
};
