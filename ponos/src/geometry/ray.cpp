#include "geometry/ray.h"

namespace ponos {

Ray2::Ray2() {}

Ray2::Ray2(const Point2 &origin, const Vector2 &direction)
    : o(origin), d(normalize(direction)) {}

Ray3::Ray3() {}

Ray3::Ray3(const Point3 &origin, const Vector3 &direction)
    : o(origin), d(normalize(direction)) {}

} // ponos namespace
