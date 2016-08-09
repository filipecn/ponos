#pragma once

#include <ponos.h>

namespace helios {

  class Shape;

  struct DifferentialGeometry {

    DifferentialGeometry();
    DifferentialGeometry(const ponos::Point3 &_p,
                         const ponos::Vector3 &_dpdu, const ponos::Vector3 &_dpdv,
                         const ponos::Normal &_dndu, const ponos::Normal &_dndv,
                         float _u, float _v, const Shape *_shape);
		ponos::point p;
		ponos::Normal n;
    float u, v;
    const Shape *shape;
		ponos::vec3 dpdu, dpdv;
		ponos::Normal dndu, dndv;
  };

} // helios namespace
