#ifndef AERGIA_HELPERS_CARTESIAN_GRID_H
#define AERGIA_HELPERS_CARTESIAN_GRID_H

#include <ponos.h>

#include "scene/scene_object.h"
#include "utils/open_gl.h"

namespace aergia {

	/* cartesian grid
	 * Represents a cartesian grid.
	 */
	class CartesianGrid : public SceneObject  {
		public:
			CartesianGrid() {}
			/* Constructor.
			 * @d **[in]** delta in all axis
			 * Creates a grid **[-d, d] x [-d, d] x [-d, d]**
			 */
			CartesianGrid(int d);
			/* Constructor.
			 * @dx **[in]** delta X
			 * @dy **[in]** delta Y
			 * @dz **[in]** delta Z
			 * Creates a grid **[-dx, dx] x [-dy, dy] x [-dz, dz]**
			 */
			CartesianGrid(int dx, int dy, int dz);
			/* set
			 * @d **[in]** dimension index (x = 0, ...)
			 * @a **[in]** lowest coordinate
			 * @b **[in]** highest coordinate
			 * Set the limits of the grid for an axis.
			 *
			 * **Example:** If we want a grid with **y** coordinates in **[-5,5]**, we call **set(1, -5, 5)**
			 */
			void setDimension(size_t d, int a, int b);
			/* @inherit */
			void draw() const override;

			ponos::Interval<int>  planes[3];
	};

} // aergia namespace

#endif // AERGIA_HELPERS_CARTESIAN_GRID_H
