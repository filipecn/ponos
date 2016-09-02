#ifndef POSEIDON_STRUCTURES_VDB_MAC_GRID_H
#define POSEIDON_STRUCTURES_VDB_MAC_GRID_H

#include "structures/vdb_grid.h"

namespace poseidon {

	/* Mac-Grid structure.
	 * Uses OpenVDB library to manipulate a staggered grid.
	 */
	class VDBMacGrid {
  	public:
			/* Constructor.
			 * @d **[in]** dimensions
			 * @b **[in]** background (default value)
			 * @s **[in]** scale
			 * @o **[in]** offset
			 */
			VDBMacGrid(const ponos::ivec3& d, const float& b, const float& s, const ponos::vec3& o);
			virtual ~VDBMacGrid() {}

			// dimensions
			ponos::ivec3 dimensions;

		private:
			openvdb::VectorGrid::Ptr v;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_VDB_MAC_GRID_H

