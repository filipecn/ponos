#include "geometry/ray.h"

namespace ponos {

  Ray::Ray()
    : min_t(0.f), max_t(INFINITY), time(0.f), depth(0) {}

  Ray::Ray(const Point3& origin, const Vector3& direction, float start,
                 float end, float t, int d)
                 : o(origin), d(direction), min_t(start),
                   max_t(end), time(t), depth(d) {}

  Ray::Ray(const Point3& origin, const Vector3& direction, const Ray& parent,
                float start, float end)
                 : o(origin), d(direction), min_t(start),
                   max_t(end), time(parent.time), depth(parent.depth + 1) {}

} // ponos namespace
