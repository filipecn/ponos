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

#ifndef PONOS_GEOMETRY_ELLIPSE_H
#define PONOS_GEOMETRY_ELLIPSE_H

#include <ponos/geometry/parametric_surface.h>
#include <ponos/geometry/shape.h>
#include <ponos/geometry/transform.h>

namespace ponos {

class Ellipse : public Shape {
public:
  Ellipse() : a(0.f), b(0.f) {}
  Ellipse(real_t _a, real_t _b, point2 center) : a(_a), b(_b), c(center) {}
  real_t a;  //!< x denominator
  real_t b;  //!< y denominator
  point2 c; //!< center
};

class ParametricEllipse final : public Ellipse,
                                public ParametricCurveInterface {
public:
  ParametricEllipse() : Ellipse() {}
  ParametricEllipse(real_t a, real_t b, point2 c) : Ellipse(a, b, c) {}
  /** Compute euclidian coordinates
   * \param t parametric param **[0, 1]**
   * \returns euclidian coordinates
   */
  point2 operator()(real_t t) const override {
    real_t angle = t * Constants::two_pi;
    return this->c + vec2(this->a * cosf(angle), this->b * sinf(angle));
  }
};
} // ponos namespace

#endif // PONOS_GEOMETRY_ELLIPSE_H
