#ifndef PONOS_GEOMETRY_SPHERE_H
#define PONOS_GEOMETRY_SPHERE_H

#include "geometry/point.h"

namespace ponos {

	class Circle {
		public:
			Circle(Point2 center, float radius)
				: c(center), r(radius) {}
			virtual ~Circle() {}

			Point2 c;
			float r;
	};

	class Sphere {
		public:
			Sphere()
			: r(0.f) {}

			Sphere(Point3 center, float radius)
			: c(center), r(radius) {}
			virtual ~Sphere() {}

			Point3 c;
			float r;
	};

} // ponos namespace

#endif
