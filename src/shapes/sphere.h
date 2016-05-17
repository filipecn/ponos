#pragma once

#include "core/shape.h"

namespace ponos {

  class Sphere : public Shape {
  public:
    Sphere(const Transform *o2w, const Transform *w2o, bool ro,
           float rad, float z0, float z1, float pm);
    BBox objectBound() const override;
    bool intersect(const Ray &r, float *tHit, float *rayEpsilon,
                   DifferentialGeometry *dg) const;
    bool intersectP(const Ray &r) const;
    float surfaceArea() const;
  private:
    float radius;
    float phiMax;
    float zmin, zmax;
    float thetaMin, thetaMax;
  };

} // ponos namespace
