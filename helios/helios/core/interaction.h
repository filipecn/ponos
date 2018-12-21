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
#ifndef HELIOS_INTERACTION_H
#define HELIOS_INTERACTION_H

#include <helios/geometry/h_ray.h>
#include <ponos/geometry/point.h>

namespace helios {

class Shape;
class Primitive;
/// Represents local information of interaction of light and elements of the
/// scene.
struct Interaction {
  Interaction() = default;
  /// \param point point of interaction
  /// \param normal if in a surface, the normal at **point**
  /// \param pointError floating point associated with point computation
  /// \param outgoingDirection ray's outgoing direction (negative direction)
  /// \param t ray's parametric coordinate
  Interaction(const ponos::point3f &point, const ponos::normal3f &normal,
              const ponos::vec3f &pointError,
              const ponos::vec3f &outgoingDirection,
              real_t t /*, const MediumInterface& mediumInterface*/);
  /// \return true if interaction with surface (normal != 0)
  bool isSurfaceInteraction() const;
  /// \param d spawn direction
  /// \return ray leaving the intersection point
  HRay spawnRay(const ponos::vec3f &d) const;
  /// \param p2 destination point
  /// \return ray leaving the intersection point
  HRay spawnRayTo(const ponos::point3f &p2) const;

  ponos::point3f p;    //!< point of interaction
  real_t time;         //!< time of interaction (ray's parametric coordinate)
  ponos::vec3f pError; //!< error associated with p computation
  ponos::vec3f wo;     //!< negative ray direction (outgoing direction)
  ponos::normal3f n;   //!< surface's normal at p (if a surface exists)
};

/// The geometry of a ray-shape interaction at a particular point on its surface
class SurfaceInteraction : public Interaction {
public:
  SurfaceInteraction() = default;
  /// \param point point of interaction
  /// \param pointError floating point associated with point computation
  /// \param uv parametric coordinated of the surface at **Interaction::p**
  /// \param outgoingDirection ray's outgoing direction (negative direction)
  /// \param dpdu partial derivative of u at p
  /// \param dpdv partial derivative of v at p
  /// \param dndu change in surface's normal at u direction
  /// \param dndv change in surface's normal at v direction
  /// \param t ray's parametric coordinate
  SurfaceInteraction(const ponos::point3f &point,
                     const ponos::vec3f &pointError, const ponos::point2f &uv,
                     const ponos::vec3f &outgoingDirection,
                     const ponos::vec3f &dpdu, const ponos::vec3f &dpdv,
                     const ponos::normal3f &dndu, const ponos::normal3f &dndv,
                     real_t t, const Shape *shape);
  /// \param dpdus partial derivative of u at p
  /// \param dpdvs partial derivative of v at p
  /// \param dndus change in surface's normal at u direction
  /// \param dndvs change in surface's normal at v direction
  /// \param orientationIsAuthoritative
  void setShadingGeometry(const ponos::vec3f &dpdus, const ponos::vec3f &dpdvs,
                          const ponos::normal3f &dndus,
                          const ponos::normal3f &dndvs,
                          bool orientationIsAuthoritative);
  ponos::point2f
      uv; //!< parametric coordinated of the surface at **Interaction::p**
  ponos::vec3f dpdu;    //!< partial derivative of u at p
  ponos::vec3f dpdv;    //!< partial derivative of v at p
  ponos::normal3f dndu; //!< change in surface's normal at u direction
  ponos::normal3f dndv; //!< change in surface's normal at v direction
  struct {
    ponos::normal3f n;
    ponos::vec3f dpdu, dpdv;
    ponos::normal3f dndu, dndv;
  } shading; //!< represents pertubations on the quantities of the interaction
             //!< (ex: bump mapping)
  const Shape *shape = nullptr;
  const Primitive *primitive = nullptr;
  // TODO BSDF* bsdf = nullptr;
  // TODO BSSRDF *bssrdf = nullptr;

};

} // namespace helios

#endif // HELIOS_INTERACTION_H
