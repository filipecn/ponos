#ifndef PONOS_GEOMETRY_SEGMENT_H
#define PONOS_GEOMETRY_SEGMENT_H

#include "geometry/point.h"

namespace ponos {

	/* Line segment **ab**
	 */
	template<typename T>
	class Segment {
  	public:
			Segment() {}
	 		Segment(T _a, T _b)
				: a(_a), b(_b) {}
			virtual ~Segment() {}

			float length() const {
				return (b - a).length();
			}

			float length2() const {
				return (b - a).length2();
			}

			T a, b;
	};

	typedef Segment<Point3> Segment3;

} // ponos namespace

#endif // PONOS_GEOMETRY_SEGMENT_H

