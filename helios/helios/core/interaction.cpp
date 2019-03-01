// Created by filipecn on 2018-12-06.
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
#include "interaction.h"
#include <helios/common/globals.h>
#include <helios/core/interaction.h>

using namespace ponos;

namespace helios {
Interaction::Interaction(
    const point3f &point, const normal3f &normal, const vec3f &pointError,
    const vec3f &outgoingDirection,
    real_t t /*TODO , const MediumInterface& mediumInterface*/)
    : p(point), time(t), pError(pointError), wo(outgoingDirection),
      n(normal) /*TODO mediumInterface(mediumInterface)*/ {}

bool Interaction::isSurfaceInteraction() const { return n != normal3(); }

HRay Interaction::spawnRay(const vec3f &d) const {
  point3f o = offsetRayOrigin(p, pError, n, d);
  return HRay(o, d, Constants::real_infinity, time /*TODO , medium(d)*/);
}

HRay Interaction::spawnRayTo(const point3f &p2) const {
  point3f origin = offsetRayOrigin(p, pError, n, p2 - p);
  vec3f d = p2 - origin;
  return HRay(origin, d, 1 - Globals::shadowEpsilon, time /*, medium(d)*/);
}

SurfaceInteraction::SurfaceInteraction(
    const point3f &point, const vec3f &pointError, const point2f &uv,
    const vec3f &outgoingDirection, const vec3f &dpdu, const vec3f &dpdv,
    const normal3f &dndu, const normal3f &dndv, real_t t, const Shape *shape)
    : Interaction(point, normal3f(normalize(cross(dpdu, dpdv))), pointError,
                  outgoingDirection, t),
      uv(uv), dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), shape(shape) {
  // initialize shading geometry from true geometry
  shading.n = n;
  shading.dpdu = dpdu;
  shading.dpdv = dpdv;
  shading.dndu = dndu;
  shading.dndv = dndv;
  // adjust normal based on orientation and handedness
  //  if(shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness))
  //  {
  //    n *= -1;
  //    shading.n *= -1;
  //  }
}

void SurfaceInteraction::setShadingGeometry(const vec3f &dpdus,
                                            const vec3f &dpdvs,
                                            const normal3f &dndus,
                                            const normal3f &dndvs,
                                            bool orientationIsAuthoritative) {
  // compute shading.n
  shading.n = normal3f(normalize(cross(dpdus, dpdvs)));
  //  if(shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness))
  //  {
  //    shading.n *= -1;
  //    if(orientationIsAuthoritative)
  //        n = Faceforward(n, shading.n);
  //    else shading.n = Faceforward(shading.n, n);
  //  }
  shading.dpdu = dpdus;
  shading.dpdv = dpdvs;
  shading.dndu = dndus;
  shading.dndv = dndvs;
}

void SurfaceInteraction::computeScatteringFunctions(const RayDifferential &ray,
                                                    ponos::MemoryArena &arena,
                                                    bool allowMultipleLobes,
                                                    TransportMode mode) {
  computeDifferentials(ray);
  primitive->computeScatteringFunctions(this, arena, mode, allowMultipleLobes);
}

} // namespace helios