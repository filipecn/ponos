#include "geometry/h_ray.h"

using namespace ponos;

namespace helios {

  HRay::HRay()
    : min_t(0.f), max_t(INFINITY), time(0.f), depth(0) {}

  HRay::HRay(const Point3& origin, const Vector3& direction, float start,
                 float end, float t, int d)
                 : Ray3(origin, direction), min_t(start),
                   max_t(end), time(t), depth(d) {}

  HRay::HRay(const Point3& origin, const Vector3& direction, const HRay& parent,
                float start, float end)
                 : Ray3(origin, direction), min_t(start),
                   max_t(end), time(parent.time), depth(parent.depth + 1) {}

} // helios namespace
