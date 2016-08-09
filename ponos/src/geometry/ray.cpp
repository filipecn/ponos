#include "geometry/ray.h"

namespace ponos {

	Ray3::Ray3() {}

	Ray3::Ray3(const Point3& origin, const Vector3& direction)
		: o(origin), d(direction) {}

} // ponos namespace
