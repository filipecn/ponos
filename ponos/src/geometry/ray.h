#ifndef PONOS_GEOMETRY_RAY_H
#define PONOS_GEOMETRY_RAY_H

#include "geometry/point.h"
#include "geometry/vector.h"

namespace ponos {

class Ray2 {
public:
  Ray2();
  Ray2(const Point2 &origin, const Vector2 &direction);
  virtual ~Ray2() {}
  Point2 operator()(float t) const { return o + d * t; }
  friend std::ostream &operator<<(std::ostream &os, const Ray2 &r) {
    os << "[Ray]\n";
    os << r.o << r.d;
    return os;
  }

  Point2 o;
  Vector2 d;
};

class Ray3 {
public:
  Ray3();
  Ray3(const Point3 &origin, const Vector3 &direction);
  virtual ~Ray3() {}
  Point3 operator()(float t) const { return o + d * t; }

  friend std::ostream &operator<<(std::ostream &os, const Ray3 &r) {
    os << "[Ray]\n";
    os << r.o << r.d;
    return os;
  }

  Point3 o;
  Vector3 d;
};

typedef Ray3 ray3;

} // ponos namespace

#endif
