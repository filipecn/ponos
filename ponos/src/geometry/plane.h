#ifndef PONOS_GEOMETRY_PLANE_H
#define PONOS_GEOMETRY_PLANE_H

#include "geometry/point.h"
#include "geometry/normal.h"

#include <iostream>

namespace ponos {

/** Implements plane equation.
 * 	Implements the equation <normal> X = <offset>.
 */
class Plane {
public:
  /// Default constructor
  Plane() { offset = 0.f; }

  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  Plane(Normal n, float o) {
    normal = n;
    offset = o;
  }

  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  Vector3 project(const Vector3 &v) {
    return v - dot(v, vec3(normal)) * vec3(normal);
  }
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  Vector3 reflect(const Vector3 &v) {
    return v - 2 * dot(v, vec3(normal)) * vec3(normal);
  }

  friend std::ostream &operator<<(std::ostream &os, const Plane &p) {
    os << "[Plane] offset " << p.offset << " " << p.normal;
    return os;
  }

  Normal normal;
  float offset;
};

} // ponos namespace

#endif
