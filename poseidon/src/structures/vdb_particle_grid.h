#ifndef POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H
#define POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H

#include "elements/particle.h"

#include <ponos.h>

#include <openvdb/openvdb.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>

#include <vector>

namespace poseidon {

	class VDBParticleGrid {
		public:
			/* Constructor.
			 * @d **[in]** dimensions
			 * @s **[in]** scale
			 * @o **[in]** offset
			 */
			VDBParticleGrid(const ponos::ivec3& d, const float& s, const ponos::vec3& o);
			virtual ~VDBParticleGrid() {}
			/* init
			 * Initialize all structures. Call after particles setup.
			 */
			void init();
			/* add particle
			 * @p **[in]** position
			 * @v **[in]** velocity
			 */
			void addParticle(const ponos::Point3& p, const ponos::vec3& v);
			/* add particle
			 * @c **[in]** cell index (index space)
			 * @p **[in]** position (voxel space)
			 * @v **[in]** velocity
			 *
			 * Add a particle inside a voxel. Voxel coordinates are the cube **[-1,1]**, where the voxel coordinate in world-space is mapped to **(0, 0, 0)** in voxel-space.
			 */
			void addParticle(const ponos::ivec3& c, const ponos::Point3& p, const ponos::vec3& v);
			/* iterate
			 * @bbox **[in]** search region (world space)
			 * @f **[in]** function called to every particle
			 *
			 * Iterates through all particles **p** that are inside **bbox** and call **f** for each one.
			 */
			void iterateNeighbours(ponos::BBox bbox, std::function<void(const Particle& p)> f);
			/* iterate
			 * @c **[in]** center cell (index space)
			 * @d **[in]** delta (half size of the bounding box)
			 * @f **[in]** function called to every particle
			 *
			 * Iterate over all particles that are inside the bounding box **[c - d, c + d]**.
			 */
			void iterateCellNeighbours(const ponos::ivec3& c, const ponos::ivec3& d, std::function<void(const size_t& id)> f);
			/* iterate
			 * @c **[in]** center cell (index space)
			 * @f **[in]** function called to every particle
			 *
			 * Iterate over all particles that are inside cell **c**.
			 */
			void iterateCell(const ponos::ivec3& c, const std::function<void(const size_t& id)>& f);

			// dimensions
			ponos::ivec3 dimensions;
			// scale
			float scale;

		private:
			std::vector<openvdb::Vec3f> positions;
			std::vector<openvdb::Vec3f> velocities;
			std::vector<int> ids;

			openvdb::tools::PointIndexGrid::Ptr pointIndexGrid;
			openvdb::tools::PointDataGrid::Ptr pointDataGrid;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H

