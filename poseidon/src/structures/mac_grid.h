#ifndef POSEIDON_STRUCTURES_MAC_GRID_H
#define POSEIDON_STRUCTURES_MAC_GRID_H

#include <ponos.h>

#include <memory.h>

namespace poseidon {

	enum class CellType { FLUID, AIR, SOLID };

	/* Mac-Grid structure.
	 * Stores a staggered grid with velocity components on faces and the rest of quantities at centers.
	 */
	template<class GridType = ponos::RegularGrid<float> >
		class MacGrid {
			public:
				virtual ~MacGrid() {}
				/* setter
				 * @d **[in]** dimensions
				 * @s **[in]** scale
				 * @o **[in]** offset
				 *
				 * Set macgrid dimensions and transform.
				 */
				void set(const ponos::ivec3& d, float s, ponos::vec3 o);
				// dimensions
				ponos::ivec3 dimensions;
				// velocity's x component
				std::shared_ptr<GridType> v_x;
				// velocity's y component
				std::shared_ptr<GridType> v_y;
				// velocity's z component
				std::shared_ptr<GridType> v_z;
				// pressure
				std::shared_ptr<GridType> p;
				// divergence
				std::shared_ptr<GridType> d;
				// cell type
				std::shared_ptr<ponos::RegularGrid<CellType> > cellType;
		};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_MAC_GRID_H

