#include <helios/core/shape.h>

using namespace ponos;

namespace helios {

uint32_t Shape::nextShapeId = 1;

Shape::Shape(const HTransform *o2w, const HTransform *w2o, bool ro)
    : objectToWorld(o2w), worldToObject(w2o), reverseOrientation(ro),
      transformSwapsHandedness(o2w->swapsHandedness()), shapeId(nextShapeId++) {
}

bounds3f Shape::worldBound() const { return (*objectToWorld)(objectBound()); }

bool Shape::intersectP(const HRay &ray, bool testAlphaTexture) const {
  real_t tHit = ray.max_t;
  SurfaceInteraction isect;
  return intersect(ray, &tHit, &isect, testAlphaTexture);
}

} // namespace helios
