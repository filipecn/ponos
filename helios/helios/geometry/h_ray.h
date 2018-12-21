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

#ifndef HELIOS_GEOMETRY_H_RAY_H
#define HELIOS_GEOMETRY_H_RAY_H

#include <ponos/ponos.h>

namespace helios {

class HRay {
public:
  HRay();
  /// \param origin
  /// \param direction
  /// \param tMax max parametric corrdinate allowed for this ray
  /// \param time current parametric coordinate (time value)
  HRay(const ponos::point3f &origin, const ponos::vec3f &direction,
       real_t tMax = ponos::Constants::real_infinity,
       real_t time = 0.f /*, const Medium* medium = nullptr*/);
  ponos::point3f operator()(real_t t) const;

  ponos::point3f o;     //!< ray's origin
  ponos::vec3f d;       //!< ray's direction
  mutable real_t max_t; //!< max parametric distance allowed
  //  const Medium* medium; //!< medium containing **o**
  real_t time; //!< current parametric distance
};

/// Spawns ray origin by offsetting along the normal. Helps to avoid
/// reintersection of rays and surfaces.
/// \param p  ray origin point
/// \param pError absolute error carried by **p** \param n normal
/// \param w spawn direction
/// \return spawned point
ponos::point3f offsetRayOrigin(const ponos::point3f &p,
                               const ponos::vec3f &pError,
                               const ponos::normal3f &n, const ponos::vec3f &w);

/* Subclass of <helios::Hray>.
 * Has 2 auxiliary rays to help algorithms like antialiasing.
 */
class RayDifferential : public HRay {
public:
  RayDifferential() { hasDifferentials = false; }
  /* Constructor.
   * @origin
   * @direction
   * @start parametric coordinate start
   * @end parametric coordinate end (how far this ray can go)
   * @t parametric coordinate
   * @d depth of recursion
   */
  explicit RayDifferential(const ponos::point3 &origin,
                           const ponos::vec3 &direction, float start = 0.f,
                           float end = INFINITY, float t = 0.f, int d = 0);
  /* Constructor.
   * @origin
   * @direction
   * @parent
   * @start parametric coordinate start
   * @end parametric coordinate end (how far this ray can go)
   */
  explicit RayDifferential(const ponos::point3 &origin,
                           const ponos::vec3 &direction, const HRay &parent,
                           float start, float end = INFINITY);

  explicit RayDifferential(const HRay &ray) : HRay(ray) {
    hasDifferentials = false;
  }

  /* Updates the differential rays
   * @s smaple spacing
   *
   * Updates the differential rays for a sample spacing <s>.
   */
  void scaleDifferentials(float s);
  bool hasDifferentials;
  ponos::point3 rxOrigin, ryOrigin;
  ponos::vec3 rxDirection, ryDirection;
};

} // namespace helios

#endif
