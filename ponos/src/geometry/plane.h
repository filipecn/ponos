#ifndef PONOS_GEOMETRY_PLANE_H
#define PONOS_GEOMETRY_PLANE_H

#include "geometry/point.h"
#include "geometry/normal.h"

#include <iostream>

namespace ponos {

	class Plane {
		public:
			Plane() {
				offset = 0.f;
			}
			Plane(ponos::Normal n, float o){
				normal = n;
				offset = o;
			}

			ponos::Normal normal;
			friend std::ostream& operator<<(std::ostream& os, const Plane& p) {
				os << "[Plane] offset " << p.offset << " " << p.normal;
				return os;
			}
			float offset;
	};

} // ponos namespace

#endif
