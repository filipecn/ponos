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

#include <ponos/algorithm/scatter.h>

namespace ponos {

void scatter(const Circle &circle, size_t n, std::vector<Point2> &points) {
  float step = PI_2 / static_cast<float>(n);
  for (size_t i = 0; i < n; i++) {
    float angle = static_cast<float>(i) * step;
    points.emplace_back(circle.c + circle.r * vec2(cosf(angle), sinf(angle)));
  }
}

void scatter(const ParametricCurveInterface *pcurve, size_t n,
             std::vector<Point2> &points) {
  for (size_t i = 0; i < n; i++) {
    points.emplace_back(
        (*pcurve)(static_cast<float>(i) / static_cast<float>(n)));
  }
}

} // ponos namespace
