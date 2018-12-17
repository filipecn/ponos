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
// Created by filipecn on 2018-12-06.
#include "interaction.h"
#include <helios/core/interaction.h>

namespace helios {
Interaction::Interaction(const ponos::point3f &point,
                         const ponos::normal3f &normal,
                         const ponos::vec3f &pointError,
                         const ponos::vec3f &outgoingDirection,
                         real_t t /*, const MediumInterface& mediumInterface*/)
    : p(point), time(t), pError(pointError), wo(outgoingDirection),
      n(normal) /*mediuInterface(mediumInterface)*/ {}

bool Interaction::isSurfaceInteraction() const { return n != ponos::normal3(); }

SurfaceInteraction::SurfaceInteraction(
    const ponos::point3f &point, const ponos::vec3f &pointError,
    const ponos::point2f &uv, const ponos::vec3f &outgoingDirection,
    const ponos::vec3f &dpdu, const ponos::vec3f &dpdv,
    const ponos::normal3f &dndu, const ponos::normal3f &dndv, real_t t,
    const Shape *shape)
    : Interaction(point,
                  ponos::normal3f(ponos::normalize(ponos::cross(dpdu, dpdv))),
                  pointError, outgoingDirection, t),
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

void SurfaceInteraction::setShadingGeometry(const ponos::vec3f &dpdus,
                                            const ponos::vec3f &dpdvs,
                                            const ponos::normal3f &dndus,
                                            const ponos::normal3f &dndvs,
                                            bool orientationIsAuthoritative) {
  // compute shading.n
  shading.n = ponos::normal3f(ponos::normalize(ponos::cross(dpdus, dpdvs)));
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

} // namespace helios