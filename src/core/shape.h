#pragma once

#include "core/differential_geometry.h"
#include "geometry/bbox.h"
#include "geometry/ray.h"
#include "geometry/transform.h"

#include <memory>
#include <vector>

namespace ponos {

  struct DifferentialGeometry;

  class Shape {
  public:
    Shape(const Transform *o2w, const Transform *w2o, bool ro);

    virtual BBox objectBound() const;
    virtual void getShadingGeometry(const Transform &o2w,
                                    const DifferentialGeometry &dg,
                                    DifferentialGeometry *dgShading) const;

    BBox worldBound() const;
    bool canIntersect() const;
    void refine(std::vector<std::shared_ptr<Shape> > &refined) const;
    bool intersect(const Ray &ray, float *tHit, float &rayEpsilon, DifferentialGeometry *gd) const;
    bool intersectP(const Ray &ray) const;
    float surfaceArea() const;

    const Transform *objectToWorld, *worldToObject;
    const bool reverseOrientation, transformSwapsHandedness;
    const uint32_t shapeId;
    static uint32_t nextShapeId;
  };

} // ponos namespace
