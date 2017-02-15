#ifndef PONOS_STRUCTURES_REGULAR_GRID_H
#define PONOS_STRUCTURES_REGULAR_GRID_H

#include "common/macros.h"
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
				RegularGrid(const ivec3& d, const T& b) {
					set(d, b);
				}
				~RegularGrid() {
					clear();
				}
				void set(const ivec3& d, const T& b) {
					clear();
					dimensions = d;
					background = b;
					bdummy = b;
					data = new T**[d[0]];
					int i;
					FOR_LOOP(i, 0, d[0])
						data[i] = new T*[d[1]];
					ponos::ivec2 dd = d.xy();
					ponos::ivec2 ij;
					FOR_INDICES0_2D(dd, ij)
						data[ij[0]][ij[1]] = new T[d[2]];
				}
				T operator()(const ivec3& i) const {
					if(i >= ivec3() && i < dimensions)
						return data[i[0]][i[1]][i[2]];
					return background;
				}
				T& operator()(const ivec3& i) {
					if(i >= ivec3() && i < dimensions)
						return data[i[0]][i[1]][i[2]];
					return bdummy;
				}
				T operator()(const int& i, const int&j, const int& k) const {
					if(	i >= 0 && i < dimensions[0] &&
							j >= 0 && i < dimensions[1] &&
							k >= 0 && i < dimensions[2])
						return data[i][j][k];
					return background;
				}
				T& operator()(const int& i, const int&j, const int& k) {
					if(	i >= 0 && i < dimensions[0] &&
							j >= 0 && i < dimensions[1] &&
							k >= 0 && i < dimensions[2])
						return data[i][j][k];
					return bdummy;
				}

			private:
				void clear() {
					ponos::ivec2 d = dimensions.xy();
					ponos::ivec2 ij;
					FOR_INDICES0_2D(d, ij)
						delete[] data[ij[0]][ij[1]];
					int i;
					FOR_LOOP(i, 0, dimensions[0])
						delete[] data[i];
					delete data;
				}

				ponos::ivec3 dimensions;
				T*** data;
				T background, bdummy;
		};


	/* Regular 2D grid
	 *
	 * Simple matrix structure.
	 */
	template<class T = int>
		class RegularGrid2D {
			public:
				RegularGrid2D()
				: data(nullptr) {}
				/* Constructor
				 * @d **[in]** dimensions
				 * @b **[in]** background (default value)
				 */
				RegularGrid2D(const ivec2& d, const T& b) {
					set(d, b);
				}
				~RegularGrid2D() {
					clear();
				}
				void set(const ivec2& d, const T& b) {
					clear();
					dimensions = d;
					background = b;
					bdummy = b;
					data = new T*[d[0]];
					int i;
					FOR_LOOP(i, 0, d[0])
						data[i] = new T[d[1]];
				}
				T operator()(const ivec2& i) const {
					if(i >= ivec2() && i < dimensions)
						return data[i[0]][i[1]];
					return background;
				}
				T& operator()(const ivec2& i) {
					if(i >= ivec2() && i < dimensions)
						return data[i[0]][i[1]];
					return bdummy;
				}
				T operator()(const int& i, const int&j, const int& k) const {
					if(	i >= 0 && i < dimensions[0] &&
							j >= 0 && i < dimensions[1])
						return data[i][j];
					return background;
				}
				T& operator()(const int& i, const int&j, const int& k) {
					if(	i >= 0 && i < dimensions[0] &&
							j >= 0 && i < dimensions[1])
						return data[i][j];
					return bdummy;
				}

			private:
				void clear() {
					if(!data) return;
					ponos::ivec2 d = dimensions;
					int i;
					FOR_LOOP(i, 0, d[0])
						delete[] data[i];
					delete data;
				}

				ponos::ivec2 dimensions;
				T** data;
				T background, bdummy;
		};


}  // ponos namespace

#endif
