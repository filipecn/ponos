#pragma once

#include <ponos.h>

namespace helios {

  class HRay : public ponos::Ray3 {
  public:
    HRay();
    HRay(const ponos::Point3& origin, const ponos::Vector3& direction,
        float start = 0.f, float end = INFINITY, float t = 0.f, int d = 0);
    HRay(const ponos::Point3& origin, const ponos::Vector3& direction, const HRay& parent,
        float start, float end = INFINITY);

		mutable float min_t, max_t;
		float time;
		int depth;
	};

} // helios namespace


