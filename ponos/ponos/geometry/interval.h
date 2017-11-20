#ifndef PONOS_GEOMETRY_INTERVAL_H
#define PONOS_GEOMETRY_INTERVAL_H

namespace ponos {

	template <class T>
		class Interval {
			public:
				Interval() {}

				T low, high;
		};

} // ponos namespace

#endif
