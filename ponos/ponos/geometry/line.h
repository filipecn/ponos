#ifndef PONOS_GEOMETRY_LINE_H
#define PONOS_GEOMETRY_LINE_H

#include <ponos/geometry/point.h>
#include <ponos/geometry/vector.h>

namespace ponos {

/// Represents a line built from a point and a vector.
class Line {
public:
  Line() = default;
  /// \param _a line point
  /// \param _d direction
  Line(Point3 _a, vec3 _d) {
    a = _a;
    d = normalize(_d);
  }
  /// \return unit vector representing line direction
  vec3 direction() const { return normalize(d); }
  /// \param t parametric coordiante
  /// \return euclidian point from parametric coordinate **t**
  Point3 operator()(float t) const { return a + d * t; }
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  float projection(Point3 p) const { return dot((p - a), d); }
  /// \param p point
  /// \return closest point in line from **p**
  Point3 closestPoint(Point3 p) const { return (*this)(projection(p)); }

  friend std::ostream &operator<<(std::ostream &os, const Line &p) {
    os << "[Line]\n";
    os << p.a << p.d;
    return os;
  }
  Point3 a; ///< line point
  vec3 d; ///< line direction
};

} // ponos namespace

#endif
