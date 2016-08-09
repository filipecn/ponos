#include "core/shape.h"

#include "core/differential_geometry.h"

using namespace ponos;

namespace helios {

  uint32_t Shape::nextShapeId = 1;

  Shape::Shape(const Transform *o2w, const Transform *w2o, bool ro)
    : objectToWorld(o2w), worldToObject(w2o), reverseOrientation(ro),
      transformSwapsHandedness(o2w->swapsHandedness()),
      shapeId(nextShapeId++) {}

  BBox Shape::worldBound() const {
    return (*objectToWorld)(objectBound());
  }

  bool Shape::canIntersect() const {
    return true;
  }

  void Shape::refine(std::vector<std::shared_ptr<Shape> > &refined) const {
    LOG << "Unimplemented Shape::refine()";
  }

  bool Shape::intersect(const HRay &ray, float *tHit, float *rayEpsilon,
                        DifferentialGeometry *gd) const {
    LOG << "Unimplemented Shape::intersect()";
    return false;
  }

  bool Shape::intersectP(const HRay &ray) const {
    LOG << "Unimplemented Shape::intersectP()";
    return false;
  }

  float Shape::surfaceArea() const {
    LOG << "Unimplemented Shape::Area()";
    return 0.;
  }

} // helios namespace
