#include "h_ray.h"
#include <helios/geometry/h_ray.h>

using namespace ponos;

namespace helios {

HRay::HRay() : max_t(ponos::Constants::real_infinity), time(0.f) {}

HRay::HRay(const ponos::point3f &origin, const ponos::vec3f &direction,
           real_t tMax, real_t time)
    : o(origin), d(direction), max_t(tMax), time(time) /*, medium(medium)*/ {}

ponos::point3f HRay::operator()(real_t t) const { return o + d * t; }
/*
RayDifferential::RayDifferential(const ponos::Point3 &origin,
                                 const ponos::Vector3 &direction, float start,
                                 float end, float t, int d)
    : HRay(origin, direction, start, end, t, d) {
  hasDifferentials = false;
}

RayDifferential::RayDifferential(const ponos::Point3 &origin,
                                 const ponos::Vector3 &direction,
                                 const HRay &parent, float start, float end)
    : HRay(origin, direction, start, end, parent.time, parent.depth + 1) {
  hasDifferentials = false;
}

void RayDifferential::scaleDifferentials(float s) {
  rxOrigin = o + (rxOrigin - o) * s;
  ryOrigin = o + (ryOrigin - o) * s;
  rxDirection = d + (rxDirection - d) * s;
  ryDirection = d + (ryDirection - d) * s;
}*/
} // namespace helios
