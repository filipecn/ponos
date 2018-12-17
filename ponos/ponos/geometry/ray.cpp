#include <ponos/geometry/ray.h>

namespace ponos {

Ray2::Ray2() {}

Ray2::Ray2(const point2 &origin, const vec2 &direction)
    : o(origin), d(normalize(direction)) {}

Ray3::Ray3() {}

Ray3::Ray3(const point3 &origin, const vec3 &direction)
    : o(origin), d(normalize(direction)) {}

} // namespace ponos
