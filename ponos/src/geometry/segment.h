#ifndef PONOS_GEOMETRY_SEGMENT_H
#define PONOS_GEOMETRY_SEGMENT_H

namespace ponos {

	/* Line segment **ab**
	 */
	template<typename T>
	class Segment {
  	public:
	 		Segment(T _a, T _b)
				: a(_a), b(_b) {}
			virtual ~Segment() {}

			float length2() {
				return (b - a).length2();
			}

			T a, b;
	};

} // ponos namespace

#endif // PONOS_GEOMETRY_SEGMENT_H

