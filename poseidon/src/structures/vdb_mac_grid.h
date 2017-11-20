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
			float u(int i, int j, int k);
			float v(int i, int j, int k);
			float w(int i, int j, int k);
			/* get
			 * @i **[in]** x component (index space)
			 * @j **[in]** x component (index space)
			 * @k **[in]** x component (index space)
			 * @return position of (i - 0.5, j, k)
			 */
			ponos::Point3 getWorldPositionX(int i, int j, int k) const;
			ponos::Point3 getWorldPositionY(int i, int j, int k) const;
			ponos::Point3 getWorldPositionZ(int i, int j, int k) const;
			void set(int i, int j, int k, ponos::vec3 v);
			void setU(int i, int j, int k, float v);
			void setV(int i, int j, int k, float v);
			void setW(int i, int j, int k, float v);

			void computeDivergence();

			float getDivergence(int i, int j, int k);

			// dimensions
			ponos::ivec3 dimensions;

		private:
			openvdb::VectorGrid::Ptr grid;
			openvdb::FloatGrid::Ptr divGrid;

			ponos::Transform toGrid;
			ponos::Transform toWorld;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_VDB_MAC_GRID_H

