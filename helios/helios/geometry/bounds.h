// Created by filipecn on 2018-12-09.
/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef HELIOS_BOUNDS_H
#define HELIOS_BOUNDS_H

#include <helios/common/e_float.h>
#include <helios/geometry/h_ray.h>
#include <ponos/geometry/bbox.h>

namespace helios {

/// Axis-aligned region of space. Useful to represent bounding volumes.
/// \tparam T data type
template <typename T> class Bounds2 : public ponos::BBox2<T> { public: };

typedef Bounds2<real_t> bounds2;
typedef Bounds2<float> bounds2f;
typedef Bounds2<double> bounds2d;
typedef Bounds2<int> bounds2i;

/// Axis-aligned region of space. Useful to represent bounding volumes.
/// \tparam T data type
template <typename T> class Bounds3 : public ponos::BBox3<T> {
public:
  Bounds3();
  explicit Bounds3(const ponos::Point3<T> &p);
  Bounds3(const ponos::Point3<T> &p1, const ponos::Point3<T> &p2);

  /// Linearly interpolates between the corners by t
  /// \param t parametric coordinates
  /// \return interpolation weights
  ponos::Point3<T> lerp(const ponos::point3f &t) const;
  /// Computes the sphere that bounds the bounding box
  /// \param center **[out]** center of the sphere
  /// \param radius **[out]** radius of the sphere
  void boundingSphere(ponos::Point3<T> &center, real_t &radius) const;
  /// Intersects a ray
  /// \param ray
  /// \param hit0 **[out | opt]** first hit
  /// \param hit1 **[out | opt]** second hit
  /// \return true if intersection exists
  bool intersect(const HRay &ray, real_t *hit0, real_t *hit1) const;
  /// Intersects a ray using a precomputed ray direction reciprocal
  /// Note: returns true if ray is entirely inside bounds, even if no
  /// intersection is found.
  /// \param ray
  /// \param invDir direction reciprocal
  /// \param dirIsNeg 1 if component is negative
  /// \return true if intersection exists
  bool intersectP(const HRay &ray, const ponos::vec3f &invDir,
                  const int dirIsNeg[3]) const;
};

typedef Bounds3<real_t> bounds3;
typedef Bounds3<float> bounds3f;
typedef Bounds3<double> bounds3d;
typedef Bounds3<int> bounds3i;

/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encopasses **b** and **p**
template <typename T>
Bounds3<T> make_union(const Bounds3<T> &b, const Bounds3<T> &p);
/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encopasses **b** and **p**
template <typename T>
Bounds3<T> make_union(const Bounds3<T> &b, const ponos::Point3<T> &p);

#include "bounds.inl"

} // namespace helios

#endif // HELIOS_BOUNDS_H
