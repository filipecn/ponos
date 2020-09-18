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

#ifndef PONOS_GEOMETRY_SURFACE_INTERFACE_H
#define PONOS_GEOMETRY_SURFACE_INTERFACE_H

#include <ponos/geometry/bbox.h>
#include <ponos/geometry/ray.h>

namespace ponos {

struct CurveRayIntersection {
  bool exists;    //!< **true** if intersection exists
  double t;       //!< parametric coordinate from ray
  point2 point;   //!< intersection point
  normal2 normal; //!< intersection normal
};

class CurveInterface {
public:
  CurveInterface() {}
  virtual ~CurveInterface() {}
  virtual point2 closestPoint(const point2 &p) const = 0;
  virtual normal2 closestNormal(const point2 &p) const = 0;
  virtual bbox2 boundingBox() const = 0;
  virtual void closestIntersection(const Ray2 &r,
                                   CurveRayIntersection *i) const = 0;
  virtual bool intersects(const Ray2 &r) const {
    CurveRayIntersection i;
    closestIntersection(r, &i);
    return i.exists;
  }
  virtual real_t closestDistance(const point2 &p) const {
    return distance(closestPoint(p), p);
  }
};

class ImplicitCurveInterface : public CurveInterface {
public:
  ImplicitCurveInterface() {}
  virtual ~ImplicitCurveInterface() {}
  virtual double signedDistance(const point2 &p) const = 0;
};

struct SurfaceRayIntersection {
  bool exists;    //!< **true** if intersection exists
  double t;       //!< parametric coordinate from ray
  point3 point;   //!< intersection point
  normal3 normal; //!< intersection normal
};

class SurfaceInterface {
public:
  SurfaceInterface() {}
  virtual ~SurfaceInterface() {}
  virtual point3 closestPoint(const point3 &p) const = 0;
  virtual normal3 closestNormal(const point3 &p) const = 0;
  virtual bbox3 boundingBox() const = 0;
  virtual void closestIntersection(const Ray3 &r,
                                   SurfaceRayIntersection *i) const = 0;
  virtual bool intersects(const Ray3 &r) const {
    SurfaceRayIntersection i;
    closestIntersection(r, &i);
    return i.exists;
  }
  virtual double closestDistance(const point3 &p) const {
    return distance(closestPoint(p), p);
  }
};

class ImplicitSurfaceInterface : public SurfaceInterface {
public:
  ImplicitSurfaceInterface() {}
  virtual ~ImplicitSurfaceInterface() {}
  virtual double signedDistance(const point3 &p) const = 0;
};

} // namespace ponos

#endif // PONOS_GEOMETRY_SURFACE_INTERFACE_H
