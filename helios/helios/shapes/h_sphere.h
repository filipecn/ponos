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

#ifndef HELIOS_SHAPES_H_SPHERE_H
#define HELIOS_SHAPES_H_SPHERE_H

#include <helios/core/shape.h>
#include <helios/geometry/bounds.h>
#include <helios/geometry/h_ray.h>

#include <ponos/ponos.h>

namespace helios {

/// Implements the parametric form:
/// x = r sin theta cos phi
/// y = r sin theta sin phi
/// z = r cos theta
/// One can limit angle values to tighten up the surface of the sphere
class HSphere : public Shape {
public:
  /// \param o2w object to world transformation
  /// \param w2o world to object transformation
  /// \param ro reverse ortientation: indicates if the surface normal directions
  /// \param radius
  /// \param zMin lower limit of z value (up axis)
  /// \param zMax upper limit of z value (up axis)
  /// \param phiMax upper limit of phi angle
  HSphere(const HTransform *o2w, const HTransform *w2o, bool ro, real_t radius,
          real_t zMin, real_t zMax, real_t phiMax);
  bounds3f objectBound() const override;
  bool intersect(const HRay &r, real_t *tHit, SurfaceInteraction *isect,
                 bool testAlphaTexture) const override;
  bool intersectP(const HRay &r, bool testAlphaTexture) const override;
  real_t surfaceArea() const override;

private:
  real_t radius;
  real_t zmin, zmax;
  real_t thetaMin, thetaMax, phiMax;
};

} // namespace helios

#endif
