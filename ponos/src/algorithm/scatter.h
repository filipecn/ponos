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

#include "common/macros.h"
#include "geometry/parametric_surface.h"
#include "geometry/sphere.h"
#include "geometry/transform.h"

#include <functional>
#include <vector>

namespace ponos {

inline Transform2D linearSpaceTransform(const BBox2D &b, size_t w, size_t h) {
  vec2 size(w - 1, h - 1);
  return translate(vec2(b.pMin)) *
         scale(b.size(0) / size.x, b.size(1) / size.y);
}

template <typename T>
void grid_scatter(const BBox2D &bbox, size_t w, size_t h,
                  std::vector<T> &points) {
  Transform2D t(bbox);
  ivec2 ij, D(w - 1, h - 1);
  FOR_INDICES0_E2D(D, ij) {
    Point2 p = t(Point2(static_cast<float>(ij[0]) / D[0],
                        static_cast<float>(ij[1]) / D[1]));
    points.emplace_back(p.x, p.y);
  }
}

inline void grid_scatter(const BBox2D &bbox, size_t w, size_t h,
                         const std::function<void(Point2)> &f) {
  Transform2D t(bbox);
  ivec2 ij, D(w - 1, h - 1);
  FOR_INDICES0_2D(D, ij) {
    Point2 p = t(Point2(static_cast<float>(ij[0]) / D[0],
                        static_cast<float>(ij[1]) / D[1]));
    f(p);
  }
}

void scatter(const Circle &circle, size_t n, std::vector<Point2> &points);

void scatter(const ParametricCurveInterface *pcurve, size_t n,
             std::vector<Point2> &points);

} // ponos namespace

#endif // PONOS_ALGORITHM_SCATTER_H
