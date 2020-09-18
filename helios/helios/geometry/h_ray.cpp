#include "h_ray.h"
#include <helios/common/utils.h>
#include <helios/geometry/h_ray.h>

using namespace ponos;

namespace helios {

HRay::HRay() : max_t(ponos::Constants::real_infinity), time(0.f) {}

HRay::HRay(const point3f &origin, const vec3f &direction, real_t tMax,
           real_t time)
    : o(origin), d(direction), max_t(tMax), time(time) /*, medium(medium)*/ {}

point3f HRay::operator()(real_t t) const { return o + d * t; }

point3f offsetRayOrigin(const ponos::point3f &p, const ponos::vec3f &pError,
                        const ponos::normal3f &n, const ponos::vec3f &w) {
  real_t d = dot(abs(n), pError);
  vec3f offset = d * vec3f(n);
  if (dot(w, n))
    offset = -offset;
  point3f po = p + offset;
  // round offset point away from p
  for (int i = 0; i < 3; i++)
    if (offset[i] < 0)
      po[i] = nextFloatUp(po[i]);
    else if (offset[i] > 0)
      po[i] = nextFloatDown(po[i]);
  return po;
}

RayDifferential::RayDifferential() { hasDifferentials = false; }

RayDifferential::RayDifferential(const point3f &origin, const vec3f &direction,
                                 real_t tMax, real_t time)
    : HRay(origin, direction, tMax, time) {
  hasDifferentials = false;
}

void RayDifferential::scaleDifferentials(float s) {
  rxOrigin = o + (rxOrigin - o) * s;
  ryOrigin = o + (ryOrigin - o) * s;
  rxDirection = d + (rxDirection - d) * s;
  ryDirection = d + (ryDirection - d) * s;
}

} // namespace helios
