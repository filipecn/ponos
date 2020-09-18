#ifndef PONOS_GEOMETRY_RAY_H
#define PONOS_GEOMETRY_RAY_H

#include <ponos/geometry/point.h>
#include <ponos/geometry/vector.h>

namespace ponos {

class Ray2 {
public:
  Ray2();
  Ray2(const point2 &origin, const vec2 &direction);
  virtual ~Ray2() = default;
  point2 operator()(float t) const { return o + d * t; }
  friend std::ostream &operator<<(std::ostream &os, const Ray2 &r) {
    os << "[Ray]\n";
    os << r.o << r.d;
    return os;
  }

  point2 o;
  vec2 d;
};

typedef Ray2 ray2;

class Ray3 {
public:
  Ray3();
  Ray3(const point3 &origin, const vec3 &direction);
  virtual ~Ray3() = default;
  point3 operator()(float t) const { return o + d * t; }

  friend std::ostream &operator<<(std::ostream &os, const Ray3 &r) {
    os << "[Ray]\n";
    os << r.o << r.d;
    return os;
  }

  point3 o;
  vec3 d;
};

typedef Ray3 ray3;

} // namespace ponos

#endif
