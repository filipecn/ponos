/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef PONOS_ALGORITHM_SCATTER_H
#define PONOS_ALGORITHM_SCATTER_H

#include <ponos/common/macros.h>
#include <ponos/geometry/parametric_surface.h>
#include <ponos/geometry/sphere.h>
#include <ponos/geometry/transform.h>

#include <functional>
#include <vector>

namespace ponos {

template <typename T> inline std::vector<T> linspace(T a, T b, int n) {
  T step = (b - a) / static_cast<T>(n - 1);
  std::vector<T> r;
  for (int i = 0; i < n; i++)
    r.emplace_back(i * step);
  return r;
}

template <typename T>
inline std::vector<T> scatter(T a, T b, int n, std::function<T(T x)> shape) {
  std::vector<T> v = linspace(a, b, n);
  for (auto &vv : v)
    vv = shape(vv);
  return v;
}

inline Transform2 linearSpaceTransform(const bbox2 &b, size_t w, size_t h) {
  vec2 size(w - 1, h - 1);
  return translate(vec2(b.lower)) *
         scale(b.size(0) / size.x, b.size(1) / size.y);
}

template <typename T>
void grid_scatter(const bbox2 &bbox, size_t w, size_t h,
                  std::vector<T> &points) {
  Transform2 t(bbox);
  ivec2 ij, D(w - 1, h - 1);
  FOR_INDICES0_E2D(D, ij) {
    point2 p = t(point2(static_cast<float>(ij[0]) / D[0],
                        static_cast<float>(ij[1]) / D[1]));
    points.emplace_back(p.x, p.y);
  }
}

inline void grid_scatter(const bbox2 &bbox, size_t w, size_t h,
                         const std::function<void(point2)> &f) {
  Transform2 t(bbox);
  ivec2 ij, D(w - 1, h - 1);
  FOR_INDICES0_2D(D, ij) {
    point2 p = t(point2(static_cast<float>(ij[0]) / D[0],
                        static_cast<float>(ij[1]) / D[1]));
    f(p);
  }
}

void scatter(const Circle &circle, size_t n, std::vector<point2> &points);

void scatter(const ParametricCurveInterface *pcurve, size_t n,
             std::vector<point2> &points);

} // namespace ponos

#endif // PONOS_ALGORITHM_SCATTER_H
