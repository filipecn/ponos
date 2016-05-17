#pragma once

#include "geometry/normal.h"
#include "geometry/point.h"
#include "geometry/vector.h"

namespace ponos {

  class Shape;

  struct DifferentialGeometry {
    DifferentialGeometry();
    DifferentialGeometry(const Point3 &_p,
                         const Vector3 &_dpdu, const Vector3 &_dpdv,
                         const Normal &_dndu, const Normal &_dndv,
                         float _u, float _v, const Shape *_shape);
    point p;
    Normal n;
    float u, v;
    const Shape *shape;
    vec3 dpdu, dpdv;
    Normal dndu, dndv;
  };

} // ponos namespace
