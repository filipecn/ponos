#ifndef PONOS_GEOMETRY_LINE_H
#define PONOS_GEOMETRY_LINE_H

#include <ponos/geometry/point.h>
#include <ponos/geometry/vector.h>

namespace ponos {

class Line {
public:
  Line() {}
  Line(Point3 _a, vec3 _d) {
    a = _a;
    d = normalize(_d);
  }

  vec3 direction() const { return normalize(d); }

  Point3 operator()(float t) const { return a + d * t; }

  float projection(Point3 p) const { return dot((p - a), d); }

  Point3 closestPoint(Point3 p) const { return (*this)(projection(p)); }

  friend std::ostream &operator<<(std::ostream &os, const Line &p) {
    os << "[Line]\n";
    os << p.a << p.d;
    return os;
  }

  Point3 a;
  vec3 d;
};

} // ponos namespace

#endif
