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
  Circle() { r = 0.f; }
  Circle(Point2 center, float radius) : c(center), r(radius) {
    this->type = ShapeType::SPHERE;
  }
  virtual ~Circle() {}

  Point2 c;
  float r;
};

class ParametricCircle final : public Circle, public ParametricCurveInterface {
public:
  ParametricCircle() { r = 0.f; }
  ParametricCircle(Point2 center, float radius) : Circle(center, radius) {}
  /** Compute euclidian coordinates
   * \param t parametric param **[0, 1]**
   * \returns euclidian coordinates
   */
  Point2 operator()(float t) const override {
    float angle = t * PI_2;
    return this->c + this->r * vec2(cosf(angle), sinf(angle));
  }
};

class ImplicitCircle final : public ImplicitCurveInterface {
public:
  ImplicitCircle() : r(0.f) {}

  ImplicitCircle(Point2 center, float radius) : c(center), r(radius) {}
  ~ImplicitCircle() {}

  Point2 closestPoint(const Point2 &p) const override {
    return c + r * vec2(this->closestNormal(p));
  }
  Normal2D closestNormal(const Point2 &p) const override {
    if (c == p)
      return Normal2D(1, 0);
    vec2 n = normalize(c - p);
    return Normal2D(n.x, n.y);
  }
  BBox2D boundingBox() const override { return BBox2D(c - r, c + r); }
  void closestIntersection(const Ray2 &r,
                           CurveRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  double signedDistance(const Point2 &p) const override {
    return distance(c, p) - r;
  }

  Point2 c;
  float r;
};

class Sphere : public SurfaceInterface {
public:
  Sphere() : r(0.f) {}

  Sphere(Point3 center, float radius) : c(center), r(radius) {}
  virtual ~Sphere() {}

  Point3 closestPoint(const Point3 &p) const override {
    return c + r * vec3(this->closestNormal(p));
  }
  Normal closestNormal(const Point3 &p) const override {
    if (c == p)
      return Normal(1, 0, 0);
    vec3 n = normalize(c - p);
    return Normal(n.x, n.y, n.z);
  }
  BBox boundingBox() const override { return BBox(c - r, c + r); }
  void closestIntersection(const Ray3 &r,
                           SurfaceRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  Point3 c;
  float r;
};

class ImplicitSphere final : public ImplicitSurfaceInterface {
public:
  ImplicitSphere() : r(0.f) {}

  ImplicitSphere(Point3 center, float radius) : c(center), r(radius) {}
  ~ImplicitSphere() {}

  Point3 closestPoint(const Point3 &p) const override {
    return c + r * vec3(this->closestNormal(p));
  }
  Normal closestNormal(const Point3 &p) const override {
    if (c == p)
      return Normal(1, 0, 0);
    vec3 n = normalize(c - p);
    return Normal(n.x, n.y, n.z);
  }
  BBox boundingBox() const override { return BBox(c - r, c + r); }
  void closestIntersection(const Ray3 &r,
                           SurfaceRayIntersection *i) const override {
    UNUSED_VARIABLE(r);
    UNUSED_VARIABLE(i);
  }
  double signedDistance(const Point3 &p) const override {
    return distance(c, p) - r;
  }

  Point3 c;
  float r;
};

inline BBox2D compute_bbox(const Circle &po, const Transform2D *t = nullptr) {
  BBox2D b;
  ponos::Point2 center = po.c;
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

} // ponos namespace

#endif
