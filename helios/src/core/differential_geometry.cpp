#include "core/differential_geometry.h"

#include "core/shape.h"

using namespace ponos;

namespace helios {

  DifferentialGeometry::DifferentialGeometry() {
    u = v = 0.;
    shape = NULL;
  }
  DifferentialGeometry::DifferentialGeometry(const Point3 &_p,
    const Vector3 &_dpdu, const Vector3 &_dpdv,
    const Normal &_dndu, const Normal &_dndv,
    float _u, float _v, const Shape *_shape)
    : p(_p), dpdu(_dpdu), dpdv(_dpdv), dndu(_dndu), dndv(_dndv) {
      n = Normal(normalize(cross(dpdu, dpdv)));
      u = _u;
      v = _v;
      shape = _shape;
      if(shape && (shape->reverseOrientation ^ shape->transformSwapsHandedness))
      n *= -1.f;
    }

  } // helios namespace
