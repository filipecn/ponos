#ifndef PONOS_STRUCTURES_C_GRID_INTERFACE_H
#define PONOS_STRUCTURES_C_GRID_INTERFACE_H

#include "geometry/point.h"
#include "geometry/transform.h"
#include "geometry/vector.h"

#include <algorithm>

namespace ponos {

	enum class CGridAccessMode {CLAMP_TO_DEDGE, BACKGROUND, REPEAT};

	/* Continuous 3D grid interface
	 *
	 * Continuous grids allow interpolation between coordinates providing access to floating point indices.
	 */
	template<typename T>
		class CGridInterface {
			public:
				virtual ~CGridInterface() {}
				/* set
				 * @i **[in]** coordinate (grid space)
				 * @v **[in]** value
				 */
				virtual void set(const ivec3& i, const T& v) = 0;
				/* get
				 * @i **[in]** coordinate (grid space)
				 * @return value at coordinate **i**
				 */
				virtual T operator()(const ivec3& i) const = 0;
				// virtual T& operator()(const ivec3& i) = 0;
				/* get
				 * @i **[in]** X coordinate (grid space)
				 * @j **[in]** Y coordinate (grid space)
				 * @k **[in]** Z coordinate (grid space)
				 * @return value at coordinate **(i, j, k)**
				 */
				virtual T operator()(const uint& i, const uint&j, const uint& k) const = 0;
				// virtual T& operator()(const uint& i, const uint&j, const uint& k) = 0;
				/* get
				 * @i **[in]** coordinate (grid space)
				 * @return value at coordinate **i**
				 */
				virtual T operator()(const vec3& i) const = 0;
				/* get
				 * @i **[in]** X coordinate (grid space)
				 * @j **[in]** Y coordinate (grid space)
				 * @k **[in]** Z coordinate (grid space)
				 * @return value at coordinate **(i, j, k)**
				 */
				virtual T operator()(const float& i, const float&j, const float& k) const = 0;
				/* test
				 * @i coordinate
				 *
				 * @return **true** if the grid contains the coordinate **i**
				 */
				bool belongs(const ivec3& i) const {
					return ivec3() <= i && i < dimensions;
				}
				// access mode
				CGridAccessMode mode;
				// default value
				T background;
				// grid dimensions
				ivec3 dimensions;
				// grid to world transform
				Transform toWorld;
				// world to grid transform
				Transform toGrid;
		};

	template<class T>
		class CGrid2DInterface {
			public:
				void setTransform(Vector2 _offset, Vector2 _cellSize) {
					offset = _offset;
					toWorld.reset();
					toWorld.scale(_cellSize.x, _cellSize.y);
					toWorld.translate(offset);
					toWorld.computeInverse();
					toGrid = inverse(toWorld);
				}

				void setDimensions(uint32_t w, uint32_t h) {
					width = w;
					height = h;
				}

				void set(uint32_t w, uint32_t h, Vector2 _offset, Vector2 _cellSize) {
					setTransform(_offset, _cellSize);
					setDimensions(w, h);
				}

				ponos::Point<int, 2> cell(ponos::Point2 wp) {
					ponos::vec2 delta(0.5f, 0.5f);
					return ponos::Point<int, 2>(toGrid(wp) + delta);
				}

				ponos::Point<int, 2> safeCell(ponos::Point2 wp) {
					ponos::vec2 delta(0.5f, 0.5f);
					ponos::Point<int, 2> gp(toGrid(wp) + delta);
					gp[0] = std::min(static_cast<int>(width) - 1, static_cast<int>(std::max(0, gp[0])));
					gp[1] = std::min(static_cast<int>(height) - 1, static_cast<int>(std::max(0, gp[1])));
					return gp;
				}

				bool belongs(ponos::Point<int, 2> c) {
					return 0 <= c[0] && c[0]< static_cast<int>(width) && 0 <= c[1] && c[1] < static_cast<int>(height);
				}

				T border;
				bool useBorder;

				uint32_t width, height;
				Vector2 offset;
				Transform2D toWorld, toGrid;
		};

} // ponos namespace

#endif
