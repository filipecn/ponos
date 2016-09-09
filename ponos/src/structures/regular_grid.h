#ifndef PONOS_STRUCTURES_REGULAR_GRID_H
#define PONOS_STRUCTURES_REGULAR_GRID_H

#include "geometry/vector.h"
#include "log/debug.h"

#include <memory>
#include <vector>

namespace ponos {

	/* Regular grid
	 *
	 * Simple matrix structure.
	 */
	template<class T = int>
		class RegularGrid {
			public:
				RegularGrid() {}
				/* Constructor
				 * @d **[in]** dimensions
				 * @b **[in]** background (default value)
				 */
				RegularGrid(const ivec3& d, const T& b);
				~RegularGrid();
				void set(const ivec3& i, const T& v) {}
				T operator()(const ivec3& i) const {}
				T& operator()(const ivec3& i) {}
				T operator()(const uint& i, const uint&j, const uint& k) const {}
				T& operator()(const uint& i, const uint&j, const uint& k) {}

			private:
				T*** data;
		};

}  // ponos namespace

#endif
