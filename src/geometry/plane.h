#pragma once

#include "geometry/point.h"
#include "geometry/normal.h"

namespace ponos {

	class Plane {
		public:
			Plane(ponos::Normal n, float o){
				normal = n;
				offset = o;
			}

			ponos::Normal normal;
			float offset;
	};

} // ponos namespace

