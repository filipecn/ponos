#pragma once

#include "geometry/line.h"
#include "geometry/plane.h"
#include "geometry/sphere.h"
#include "geometry/utils.h"
#include "geometry/vector.h"

namespace ponos {

	bool plane_line_intersection(const Plane pl, const Line l, Point3& p);
	bool sphere_line_intersection(const Sphere s, const Line l, Point3& p1, Point3& p2);

} // ponos namespace
