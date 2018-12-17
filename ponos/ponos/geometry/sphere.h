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

#ifndef PONOS_GEOMETRY_SPHERE_H
#define PONOS_GEOMETRY_SPHERE_H

#include <ponos/geometry/parametric_surface.h>
#include <ponos/geometry/shape.h>
#include <ponos/geometry/transform.h>

namespace ponos {

class Circle : public Shape {
public:
  Circle() { r = 0; }
  Circle(point2 center, real_t radius) : c(center), r(radius) {
    this->type = ShapeType::SPHERE;
  }
  virtual ~Circle() {}

  point2 c;
  real_t r;
};

class ParametricCircle final : public Circle, public ParametricCurveInterface {
public:
  ParametricCircle() { r = 0.f; }
  ParametricCircle(point2 center, real_t radius) : Circle(center, radius) {}
  /** Compute euclidian coordinates
   * \param t parametric param **[0, 1]**
   * \returns euclidian coordinates
   */
  point2 operator()(real_t t) const override {
    real_t angle = t * Constants::two_pi;
    return this->c + this->r * vec2(cosf(angle), sinf(angle));
  }
};

class ImplicitCircle final : public ImplicitCurveInterface {
public:
  ImplicitCircle() : r(0.f) {}

  ImplicitCircle(point2 center, real_t radius) : c(center), r(radius) {}
  ~ImplicitCircle() {}

  point2 closestPoint(const point2 &p) const override {
    return c + r * vec2(this->closestNormal(p));
  }
  normal2 closestNormal(const point2 &p) const override {
    if (c == p)
      return normal2(1, 0);
    vec2 n = normalize(c - p);
    return normal2(n.x, n.y);
  }
  bbox2 boundingBox() const override { return bbox2(c - r, c + r); }
  void closestIntersection(const Ray2 &r,
                           CurveRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  double signedDistance(const point2 &p) const override {
    return distance(c, p) - r;
  }

  point2 c;
  real_t r;
};

class Sphere : public SurfaceInterface {
public:
  Sphere() : r(0.f) {}

  Sphere(point3 center, real_t radius) : c(center), r(radius) {}
  virtual ~Sphere() {}

  point3 closestPoint(const point3 &p) const override {
    return c + r * vec3(this->closestNormal(p));
  }
  normal3 closestNormal(const point3 &p) const override {
    if (c == p)
      return normal3(1, 0, 0);
    vec3 n = normalize(c - p);
    return normal3(n.x, n.y, n.z);
  }
  bbox3 boundingBox() const override { return bbox3(c - r, c + r); }
  void closestIntersection(const Ray3 &r,
                           SurfaceRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  point3 c;
  real_t r;
};

class ImplicitSphere final : public ImplicitSurfaceInterface {
public:
  ImplicitSphere() : r(0.f) {}

  ImplicitSphere(point3 center, real_t radius) : c(center), r(radius) {}
  ~ImplicitSphere() {}

  point3 closestPoint(const point3 &p) const override {
    return c + r * vec3(this->closestNormal(p));
  }
  normal3 closestNormal(const point3 &p) const override {
    if (c == p)
      return normal3(1, 0, 0);
    vec3 n = normalize(c - p);
    return normal3(n.x, n.y, n.z);
  }
  bbox3 boundingBox() const override { return bbox3(c - r, c + r); }
  void closestIntersection(const Ray3 &r,
                           SurfaceRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  double signedDistance(const point3 &p) const override {
    return distance(c, p) - r;
  }

  point3 c;
  real_t r;
};

inline bbox2 compute_bbox(const Circle &po, const Transform2 *t = nullptr) {
  bbox2 b;
  ponos::point2 center = po.c;
  if (t != nullptr) {
    b = make_union(b, (*t)(center + ponos::vec2(po.r, 0)));
    b = make_union(b, (*t)(center + ponos::vec2(-po.r, 0)));
    b = make_union(b, (*t)(center + ponos::vec2(0, po.r)));
    b = make_union(b, (*t)(center + ponos::vec2(0, -po.r)));
  } else {
    b = make_union(b, center + ponos::vec2(po.r, 0));
    b = make_union(b, center + ponos::vec2(-po.r, 0));
    b = make_union(b, center + ponos::vec2(0, po.r));
    b = make_union(b, center + ponos::vec2(0, -po.r));
  }
  return b;
}

} // namespace ponos

#endif
