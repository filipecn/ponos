#include "geometry/intersections.h"

namespace ponos {

	bool plane_line_intersection(const Plane pl, const Line l, Point3& p) {
		vec3 nvector = vec3(pl.normal.x, pl.normal.y, pl.normal.z);
		float k = dot(nvector, l.direction());
		if(IS_ZERO(k))
			return false;
		float r = (pl.offset - dot(nvector, vec3(l.a.x, l.a.y, l.a.z))) / k;
		p = l(r);
		return true;
	}

	bool sphere_line_intersection(const Sphere s, const Line l, Point3& p1, Point3& p2) {
			float a = dot(l.direction(), l.direction());
			float b = 2.f * dot(l.direction(), l.a - s.c);
			float c = dot(l.a - s.c, l.a - s.c) - s.r * s.r;

			float delta = b * b - 4 * a * c;
			if(delta < 0)
				return false;

			float d = sqrt(delta);

			p1 = l((-b + d) / (2.0 * a));
			p2 = l((-b - d) / (2.0 * a));
			return true;
	}


} // ponos namespace
