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

#include <ponos/geometry/point.h>
#include <ponos/geometry/surface.h>
#include <ponos/geometry/vector.h>

#include <iostream>

namespace ponos {

/** Implements the equation normal X = offset.
 */
class Plane {
public:
  /// Default constructor
  Plane() { offset = 0; }
  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  Plane(normal3 n, real_t o) {
    normal = n;
    offset = o;
  }
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,0,1)
  static Plane XY(bool invert_normal = false) { return {normal3(0, 0, 1), 0}; }
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,1,0)
  static Plane XZ(bool invert_normal = false) { return {normal3(0, 1, 0), 0}; }
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(1,0,0)
  static Plane YZ(bool invert_normal = false) { return {normal3(1, 0, 0), 0}; }

  point3 closestPoint(const point3 &p) const {
    float t = (dot(vec3(normal), vec3(p)) - offset) / vec3(normal).length2();
    return p - t * vec3(normal);
  }
  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  vec3 project(const vec3 &v) { return ponos::project(v, normal); }
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  vec3 reflect(const vec3 &v) { return ponos::reflect(v, normal); }
  bool onNormalSide(const point3 &p) const {
    return dot(vec3(normal), p - closestPoint(p)) >= 0;
  }

  friend std::ostream &operator<<(std::ostream &os, const Plane &p) {
    os << "[Plane] offset " << p.offset << " " << p.normal;
    return os;
  }

  normal3 normal;
  real_t offset;
};

/** Implements the equation normal X = offset.
 */
class ImplicitPlane2D : public ImplicitCurveInterface {
public:
  /// Default constructor
  ImplicitPlane2D() { offset = 0.f; }
  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  ImplicitPlane2D(normal2 n, real_t o) {
    normal = n;
    offset = o;
  }
  ImplicitPlane2D(point2 p, normal2 n) {
    normal = n;
    offset = dot(vec2(normal), vec2(p));
  }
  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  vec2 project(const vec2 &v) { return ponos::project(v, normal); }
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  vec2 reflect(const vec2 &v) { return ponos::reflect(v, normal); }
  point2 closestPoint(const point2 &p) const override {
    real_t t = (dot(vec2(normal), vec2(p)) - offset) / vec2(normal).length2();
    return p - t * vec2(normal);
  }
  normal2 closestNormal(const point2 &p) const override {
    if (dot(vec2(normal), vec2(p)) < 0.f)
      return -normal;
    return normal;
  }
  bbox2 boundingBox() const override { return bbox2(); }
  void closestIntersection(const Ray2 &r,
                           CurveRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  double signedDistance(const point2 &p) const override {
    return (dot(vec2(p), vec2(normal)) - offset) / vec2(normal).length();
  }

  friend std::ostream &operator<<(std::ostream &os, const ImplicitPlane2D &p) {
    os << "[Plane] offset " << p.offset << " " << p.normal;
    return os;
  }

  normal2 normal;
  real_t offset;
};

} // namespace ponos

#endif
