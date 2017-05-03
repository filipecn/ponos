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

#ifndef PONOS_GEOMETRY_PLANE_H
#define PONOS_GEOMETRY_PLANE_H

#include "geometry/point.h"
#include "geometry/normal.h"

#include <iostream>

namespace ponos {

/** Implements the equation <normal> X = <offset>.
 */
class Plane {
public:
  /// Default constructor
  Plane() { offset = 0.f; }
  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  Plane(Normal n, float o) {
    normal = n;
    offset = o;
  }
  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  Vector3 project(const Vector3 &v) { return ponos::project(v, normal); }
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  Vector3 reflect(const Vector3 &v) { return ponos::reflect(v, normal); }

  friend std::ostream &operator<<(std::ostream &os, const Plane &p) {
    os << "[Plane] offset " << p.offset << " " << p.normal;
    return os;
  }

  Normal normal;
  float offset;
};

} // ponos namespace

#endif
