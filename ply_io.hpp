// ply_io.hpp - PLY loading (happly) and natural-sorted folder listing.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <happly.h>

// Vertex positions of a PLY file (float or double x/y/z), non-finite points removed.
// If `times` is given and the vertices carry a float/double property named
// t, time or timestamp, it is returned per point (raw values); else left empty.
inline std::vector<Eigen::Vector3d> loadPly(const std::string &path,
                                            std::vector<double> *times = nullptr) {
    happly::PLYData ply(path);
    const auto v = ply.getVertexPositions();  // float or double x/y/z
    std::vector<double> t;
    if (times) {
        happly::Element &vertex = ply.getElement("vertex");
        for (const char *name : {"t", "time", "timestamp"}) {
            if (!vertex.hasProperty(name)) continue;
            try {
                t = vertex.getProperty<double>(name);
            } catch (const std::exception &) {  // not a float/double property
            }
            break;
        }
        if (t.size() != v.size()) t.clear();
        times->clear();
        times->reserve(t.size());
    }
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const auto &p = v[i];
        if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
            pts.emplace_back(p[0], p[1], p[2]);
            if (!t.empty()) times->push_back(t[i]);
        }
    }
    return pts;
}

// "frame_2" < "frame_10": digit runs compare by value.
inline bool naturalLess(const std::string &a, const std::string &b) {
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

inline std::vector<std::string> listPly(const std::filesystem::path &dir) {
    std::vector<std::string> files;
    for (const auto &e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".ply") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end(), naturalLess);
    return files;
}
