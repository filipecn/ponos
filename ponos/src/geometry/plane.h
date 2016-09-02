#ifndef PONOS_GEOMETRY_PLANE_H
#define PONOS_GEOMETRY_PLANE_H

#include "geometry/point.h"
#include "geometry/normal.h"

#include <iostream>

namespace ponos {

	/* Implements plane equation.
	 * Implements the equation <normal> X = <offset>.
	 */
	class Plane {
		public:
			Plane() {
				offset = 0.f;
			}

			/* Constructor
			 * @n **[in]** normal
			 * @o **[in]** offset
			 */
			Plane(ponos::Normal n, float o){
				normal = n;
				offset = o;
			}

			friend std::ostream& operator<<(std::ostream& os, const Plane& p) {
				os << "[Plane] offset " << p.offset << " " << p.normal;
				return os;
			}

			ponos::Normal normal;
			float offset;
	};

} // ponos namespace

#endif
