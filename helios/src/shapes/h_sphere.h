#ifndef HELIOS_SHAPES_H_SPHERE_H
#define HELIOS_SHAPES_H_SPHERE_H

#include "core/shape.h"
#include "geometry/h_ray.h"

#include <ponos.h>

namespace helios {

  class HSphere : public Shape {
  public:
    HSphere(const ponos::Transform *o2w, const ponos::Transform *w2o, bool ro,
           float rad, float z0, float z1, float pm);
		ponos::BBox objectBound() const override;
    bool intersect(const HRay &r, float *tHit, float *rayEpsilon,
                   DifferentialGeometry *dg) const;
    bool intersectP(const HRay &r) const;
    float surfaceArea() const;
  private:
    float radius;
    float phiMax;
    float zmin, zmax;
    float thetaMin, thetaMax;
  };

} // helios namespace

#endif
