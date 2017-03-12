#ifndef PONOS_GEOMETRY_SPHERE_H
#define PONOS_GEOMETRY_SPHERE_H

#include "geometry/bbox.h"
#include "geometry/point.h"
#include "geometry/shape.h"
#include "geometry/transform.h"

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

class Sphere {
public:
  Sphere() : r(0.f) {}

  Sphere(Point3 center, float radius) : c(center), r(radius) {}
  virtual ~Sphere() {}

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
