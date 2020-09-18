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

#ifndef PONOS_GEOMETRY_LINE_H
#define PONOS_GEOMETRY_LINE_H

#include <ponos/geometry/point.h>
#include <ponos/geometry/vector.h>

namespace ponos {

/// Represents a line built from a point and a vector.
class Line {
public:
  Line() = default;
  /// \param _a line point
  /// \param _d direction
  Line(point3 _a, vec3 _d) {
    a = _a;
    d = normalize(_d);
  }
  /// \return unit vector representing line direction
  vec3 direction() const { return normalize(d); }
  /// \param t parametric coordiante
  /// \return euclidian point from parametric coordinate **t**
  point3 operator()(float t) const { return a + d * t; }
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  float projection(point3 p) const { return dot((p - a), d); }
  /// \param p point
  /// \return closest point in line from **p**
  point3 closestPoint(point3 p) const { return (*this)(projection(p)); }

  friend std::ostream &operator<<(std::ostream &os, const Line &p) {
    os << "[Line]\n";
    os << p.a << p.d;
    return os;
  }
  point3 a; ///< line point
  vec3 d; ///< line direction
};

} // ponos namespace

#endif
