#pragma once

#include "core/shape.h"
#include "geometry/normal.h"
#include "geometry/point.h"
#include "geometry/vector.h"

namespace ponos {

  class Shape;

  struct DifferentialGeometry {
    DifferentialGeometry() {
      u = v = 0.;
      shape = NULL;
    }

    point p;
    Normal n;
    float u, v;
    const Shape *shape;
    vec3 dpdu, dpdv;
    Normal dndu, dndv;
  };

} // ponos namespace
