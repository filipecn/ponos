#ifndef POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H
#define POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H

#include "elements/particle.h"

#include <ponos.h>

#include <openvdb/openvdb.h>

#include <functional>
#include <vector>

namespace poseidon {

	/* spatial structure
	 *
	 * Structure for storing particles using OpenVDB library for fast neighbour search.
	 *
	 * The class names 4 different spaces: **world space**, **grid space**, **index space** and **voxel space**.
	 *
	 * **world space**: world space
	 *
	 * **grid space**: the continuous coordinates that go from (0, 0, 0) to dimensions
	 *
	 * **index space**: the lattice points of the grid
	 *
	 * **voxel space**: local coordinates of a cell, [-0.5, 0.5] where the center of the cell is the origin
	 */
	class VDBParticleGrid {
		public:
			/* Constructor.
			 * @d **[in]** dimensions
			 * @s **[in]** voxel size
			 * @o **[in]** offset
			 */
			VDBParticleGrid(const ponos::ivec3& d, const float& s, const ponos::vec3& o);
			virtual ~VDBParticleGrid() {}
			/* init
			 * Initialize all structures. Call after particles setup.
			 */
			void init();
			/* add particle
			 * @p **[in]** particle
			 */
			void addParticle(Particle *p);
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
			 * Add a particle inside a voxel. Voxel coordinates are the cube **[-0.5,0.5]**,
			 * where the voxel coordinate in world-space is mapped to **(0, 0, 0)** in voxel-space.
			 */
			void addParticle(const ponos::ivec3& c, const ponos::Point3& p, const ponos::vec3& v);
			/* set
			 * @id **[in]** particle id
			 * @p **[in]** new position
			 * @v **[in]** new velocity
			 *
			 * Updates particle **id** fields.
			 */
			void setParticle(int id, const ponos::Point3& p, const ponos::vec3& v);
			/* get
			 * @ijk **[in]** cell index (index space)
			 *
			 * @return number of particles in cell **ijk**.
			 */
			int particleCount(const ponos::ivec3& ijk);
			/* iterate
			 * @bbox **[in]** search region (world space)
			 * @f **[in]** function called to every particle
			 *
			 * Iterates through all particles **p** that are inside **bbox** and call **f** for each one.
			 */
			void iterateNeighbours(ponos::BBox bbox, std::function<void(const Particle& p)> f);
			/* iterate
			 * @center **[in]** search region center (world space)
			 * @radius **[in]** search region radius
			 * @f **[in]** function called to every particle
			 *
			 * Iterates through all particles **p** that are inside the sphere of **center** point and **radius** and call **f** for each one.
			 */
			void iterateNeighbours(ponos::Point3 center, float radius, std::function<void(const Particle& p)> f);
			/* iterate
			 * @c **[in]** center cell (index space)
			 * @d **[in]** delta (half size of the bounding box)
			 * @f **[in]** function called to every particle
			 *
			 * Iterate over all particles that are inside the bounding box **[c - d, c + d]**.
			 * **Note**: cells that are out the range of the grid dimensions are not counted,
			 * even if they contain particles.
			 */
			void iterateCellNeighbours(const ponos::ivec3& c, const ponos::ivec3& d,
																 std::function<void(const size_t& id)> f);
			/* iterate
			 * @c **[in]** center cell (index space)
			 * @f **[in]** function called to every particle
			 *
			 * Iterate over all particles that are inside cell **c**.
			 */
			void iterateCell(const ponos::ivec3& c, const std::function<void(const size_t& id)>& f);
			/* count
			 *
			 * @return number of particles currently added to the grid.
			 */
			int particleCount();
			/* transform
			 * @wp **[in]** point (world space)
			 * @return cell index that contains **wp**.
			 */
			ponos::ivec3 worldToIndex(const ponos::Point3& wp);
			/* transform
			 * @gp **[in]** point (grid space)
			 * @return cell index that contains **gp**.
			 */
			ponos::ivec3 gridToIndex(const ponos::Point3& gp);
			/* transform
			 * @wp **[in]** point (world space)
			 * @return point **wp** mapped to voxel coordinates.
			 */
			ponos::Point3 worldToVoxel(const ponos::Point3& wp);
			/* transform
			 * @gp **[in]** point (grid space)
			 * @return point **gp** mapped to voxel coordinates.
			 */
			ponos::Point3 gridToVoxel(const ponos::Point3& gp);
			/* transform
			 * @i **[in]** point (index space)
			 * @return point **i** mapped to world coordinates.
			 */
			ponos::Point3 indexToWorld(const ponos::ivec3& i);
			/* get
			 * @ijk **[in]** point (index space)
			 * @v **[out]** 8 vertices (vertices are enumerated following the permutations of x first, y and z.
			 */
			void cellVertices(const ponos::ivec3& ijk, std::vector<ponos::Point3>& v);
			/* update
			 * @d **[in]** fluid density
			 * @md **[in]** maximum density
			 * Update particles densities.
			 * @return the maximum density found
			 */
			float computeDensity(float d, float md);
			/* get
			 * @return particle list
			 */
			const std::vector<Particle*>& getParticles() const;
			/* gather
			 * @a **[in]** attribute
			 * @p **[in]** center (world coordinates)
			 * @r **[in | optional]** radius (default value is **1.5 * voxelSize**
			 * Computes weighted average value of particles atttribute values within
			 * **r** distance from **p**.
			 * @return gathered value
			 */
			float gather(ParticleAttribute a, const ponos::Point3& p, float r = -1) const;
			/* compute
			 * @ijk **[in]** coordinate (index space)
			 * @d **[in]** material density
			 * @t **[in]** particle type
			 * @return Signed Distance Field value computed from particles of type **t** inside cell **ijk**.
			 */
			float cellSDF(const ponos::ivec3& ijk, float d, ParticleType t);

			// dimensions
			ponos::ivec3 dimensions;
			// particle mass
			float particleMass;
			// voxel size
			float voxelSize;
			// world to grid transform
			ponos::Transform toWorld;
			// grid to world transform
			ponos::Transform toGrid;

		private:

			class PointList {
				public:
					typedef openvdb::Vec3R  PosType;

					PointList(const std::vector<Particle*>& p)
						: points(p) {}

					size_t size() const {
						return points.size();
					}

					void getPos(size_t n, PosType& xyz) const {
						xyz = PosType(points[n]->position.x,
								points[n]->position.y,
								points[n]->position.z);
					}

				protected:
					const std::vector<Particle*>& points;
			};

			template<typename T>
				struct WeightedAverageAccumulator {
					typedef T ValueType;
					WeightedAverageAccumulator(const std::vector<Particle*>& p, const T radius, ParticleAttribute a)
						: points(p), invRadius(1.0 / radius), weightSum(0.0), valueSum(0.0), attribute(a) {}

					void reset() { weightSum = valueSum = T(0.0); }

					void operator()(const T distSqr, const size_t pointIndex) {
						float w = points[pointIndex]->mass * ponos::sharpen(distSqr, 1.4f);
						float value = 0.f;
						switch(attribute) {
							case ParticleAttribute::VELOCITY_X: value = points[pointIndex]->velocity.x; break;
							case ParticleAttribute::VELOCITY_Y: value = points[pointIndex]->velocity.y; break;
							case ParticleAttribute::VELOCITY_Z: value = points[pointIndex]->velocity.z; break;
							case ParticleAttribute::DENSITY: value = points[pointIndex]->density; break;
							default: break;
						}
						weightSum += w;
						valueSum += w * value;
					}

					T result() const { return weightSum > T(0.0) ? valueSum / weightSum : T(0.0); }

					private:
					const std::vector<Particle*>& points;
					const T invRadius;
					T weightSum, valueSum;
					ParticleAttribute attribute;
				};

			bool updated;

			PointList *pl;
			std::vector<Particle*> particles;
			std::vector<float> densities;
			std::vector<openvdb::Vec3f> positions;
			std::vector<openvdb::Vec3f> velocities;
			std::vector<int> ids;

			openvdb::math::Transform::Ptr transform;
			openvdb::tools::PointIndexGrid::Ptr pointGridPtr;
			openvdb::tools::PointIndexFilter<PointList> *filter;

			openvdb::tools::PointIndexGrid::Ptr pointIndexGrid;
			openvdb::tools::PointDataGrid::Ptr pointDataGrid;
			openvdb::tools::AttributeSet::Descriptor::Ptr descriptor;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_VDB_PARTICLE_GRID_H

