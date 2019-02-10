/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H
#define HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H

#include <helios/geometry/h_ray.h>
#include <helios/geometry/h_transform.h>
#include <ponos/ponos.h>

namespace helios {

class AnimatedTransform {
public:
  AnimatedTransform(const HTransform *t1, real_t time1, const HTransform *t2,
                    real_t time2);
  virtual ~AnimatedTransform();

  void decompose(const ponos::mat4 &m, ponos::vec3 *T, ponos::Quaternion *Rquat,
                 ponos::mat4 *s);

  void interpolate(float time, ponos::Transform *t) const {
    // handle boundary conditions for matrix interpolation
    // float dt = (time - startTime) / (endTime - startTime);
    // interpolate translation at dt
    // interpolate rotation at dt
    // interpolate scale at dt
    // compute interpolated matrix as product of interpolated components
  }

  void operator()(const HRay &r, HRay *ret) const {
    // TODO
  }

private:
  const real_t startTime, endTime;
  const HTransform *startTransform, *endTransform;
  const bool actuallyAnimated;
  ponos::vec3 T[2];
  ponos::quat R[2];
  ponos::mat4 S[2];
  bool hasRotation;
};

} // namespace helios

#endif // HELIOS_GEOMETRY_ANIMATED_TRANSFORM_H
