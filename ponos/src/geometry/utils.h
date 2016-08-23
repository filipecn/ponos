#ifndef PONOS_GEOMETRY_UTILS_H
#define PONOS_GEOMETRY_UTILS_H

#include "geometry/vector.h"

namespace ponos {

	inline void makeCoordinateSystem(const vec3& v1, vec3* v2, vec3* v3) {
		if(fabsf(v1.x) > fabsf(v1.y)) {
			float invLen = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
			*v2 = vec3(-v1.z * invLen, 0.f, v1.x * invLen);
		}
		else {
			float invLen = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
			*v2 = vec3(0.f, v1.z * invLen, -v1.y * invLen);
		}
		*v3 = cross(v1, *v2);
	}

} // ponos namespace

#endif
