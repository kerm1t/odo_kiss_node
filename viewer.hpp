// viewer.hpp - minimal SDL2 + OpenGL 3.3 core viewer: accumulated point cloud,
// ego trajectory and current pose axes. Header-only, no GL loader needed on
// Linux / macOS.
//
//   left drag   orbit            F    follow ego on/off
//   right drag  pan              R    reset view
//   wheel       zoom             +/-  point size
//   Esc / Q     quit
//
// World is z-up. Points are (x, y, z, s) with s = scalar used for colouring.
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
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

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

        makeBuffer(vao_pts_, vbo_pts_);
        makeBuffer(vao_traj_, vbo_traj_);
        makeBuffer(vao_axes_, vbo_axes_);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        resetView();
        return true;
    }

    // Append points, 4 floats each: world x, y, z + colour scalar.
    void addPoints(const std::vector<float> &xyzs) {
        if (xyzs.empty()) return;
        if (!range_set_) setRangeFrom(xyzs);
        const size_t old_bytes = pts_.size() * sizeof(float);
        pts_.insert(pts_.end(), xyzs.begin(), xyzs.end());
        const size_t bytes = pts_.size() * sizeof(float);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts_);
        if (bytes > cap_bytes_) {  // grow x2 and re-upload everything
            cap_bytes_ = std::max(bytes, 2 * cap_bytes_);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(cap_bytes_), nullptr,
                         GL_DYNAMIC_DRAW);
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(bytes), pts_.data());
        } else {
            glBufferSubData(GL_ARRAY_BUFFER, static_cast<GLintptr>(old_bytes),
                            static_cast<GLsizeiptr>(bytes - old_bytes),
                            pts_.data() + old_bytes / sizeof(float));
        }
    }

    // Append an ego pose (sensor in world). Extends the trajectory.
    void addPose(const Eigen::Matrix4f &T) {
        pose_ = T;
        has_pose_ = true;
        traj_.insert(traj_.end(), {T(0, 3), T(1, 3), T(2, 3), 0.0f});
        glBindBuffer(GL_ARRAY_BUFFER, vbo_traj_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(traj_.size() * sizeof(float)),
                     traj_.data(), GL_DYNAMIC_DRAW);
    }

    void setStatus(const std::string &s) {
        status_ = s;
        title_dirty_ = true;
    }

    size_t numPoints() const { return pts_.size() / 4; }

    // Handle input, draw, swap. Returns false once the user wants to quit.
    bool frame() {
        if (!handleEvents()) return false;
        if (title_dirty_) {
            SDL_SetWindowTitle(win_, (title_ + "  |  " + status_ + (follow_ ? "  |  follow" : ""))
                                         .c_str());
            title_dirty_ = false;
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
        glBindVertexArray(vao_pts_);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pts_.size() / 4));

        // trajectory + ego axes on top
        glDisable(GL_DEPTH_TEST);
        const GLsizei n_traj = static_cast<GLsizei>(traj_.size() / 4);
        glBindVertexArray(vao_traj_);
        glUniform4f(u_flat_, 1.0f, 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_LINE_STRIP, 0, n_traj);
        glUniform1f(u_psize_, 4.0f);
        glDrawArrays(GL_POINTS, 0, n_traj);
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
        glBindVertexArray(0);
        SDL_GL_SwapWindow(win_);

        // cap at ~60 fps in case vsync is off, so the odometry thread keeps its cores
        const Uint32 dt = SDL_GetTicks() - last_tick_;
        if (dt < 16) SDL_Delay(16 - dt);
        last_tick_ = SDL_GetTicks();
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
    }

    bool handleEvents() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    return false;
                case SDL_KEYDOWN:
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
                    break;
                case SDL_MOUSEMOTION: {
                    const float dx = static_cast<float>(e.motion.xrel);
                    const float dy = static_cast<float>(e.motion.yrel);
                    if (e.motion.state & SDL_BUTTON_LMASK) {
                        yaw_ -= 0.005f * dx;
                        pitch_ = std::clamp(pitch_ + 0.005f * dy, -1.55f, 1.55f);
                    } else if (e.motion.state & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) {
                        // pan in the ground plane, like dragging a map
                        const Eigen::Vector3f fwd(-std::cos(yaw_), -std::sin(yaw_), 0.0f);
                        const Eigen::Vector3f right(fwd.y(), -fwd.x(), 0.0f);
                        target_ += (fwd * dy - right * dx) * dist_ * 0.0015f;
                        if (follow_) title_dirty_ = true;
                        follow_ = false;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return true;
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
out vec3 col;
void main() {
    gl_Position = mvp * vec4(a.xyz, 1.0);
    gl_PointSize = psize;
    float t = clamp((a.w - range.x) / (range.y - range.x), 0.0, 1.0);
    t = mix(0.12, 0.88, t);
    vec3 jet = clamp(1.5 - abs(4.0 * t - vec3(3.0, 2.0, 1.0)), 0.0, 1.0);
    col = flat_col.a > 0.0 ? flat_col.rgb : jet;
})";
        static const char *fs = R"(#version 330 core
in vec3 col;
out vec4 frag;
void main() { frag = vec4(col, 1.0); })";
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
    Uint32 last_tick_ = 0;

    GLuint prog_ = 0;
    GLint u_mvp_ = -1, u_range_ = -1, u_flat_ = -1, u_psize_ = -1;
    GLuint vao_pts_ = 0, vbo_pts_ = 0, vao_traj_ = 0, vbo_traj_ = 0, vao_axes_ = 0, vbo_axes_ = 0;

    std::vector<float> pts_;   // x y z s
    std::vector<float> traj_;  // x y z 0
    size_t cap_bytes_ = 0;
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
