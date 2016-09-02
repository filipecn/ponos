#include "structures/vdb_particle_grid.h"

using namespace openvdb;
using namespace openvdb::tools;

namespace poseidon {

	VDBParticleGrid::VDBParticleGrid(const ponos::ivec3& d, const float& s, const ponos::vec3& o)
		: dimensions(d), scale(s) {
			openvdb::initialize();
			openvdb::points::initialize();
		}

	void VDBParticleGrid::init() {
		openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(scale);
		// Create a PointPartitioner-compatible point list using the std::vector wrapper provided
		const PointAttributeVector<openvdb::Vec3f> pointList(positions);
		pointIndexGrid = createPointIndexGrid<PointIndexGrid>(pointList, *transform);

		// Create the PointDataGrid, position attribute is mandatory
		pointDataGrid = createPointDataGrid<PointDataGrid>(*pointIndexGrid, pointList, TypedAttributeArray<openvdb::Vec3f>::attributeType(), *transform);

		// Retrieve an iterator pointing to the first leaf node
		PointDataTree::LeafCIter iter = pointDataGrid->tree().cbeginLeaf();

		// No leaf nodes means no points, best to always check before dereferencing
		if (!iter)  std::cout << "No Points" << std::endl;

		// Add a new velocity attribute
		AttributeSet::Util::NameAndType velocityAttribute("velocity", TypedAttributeArray<openvdb::Vec3f>::attributeType());
		appendAttribute(pointDataGrid->tree(), velocityAttribute);

		// Create a point attribute list using the std::vector wrapper provided
		const PointAttributeVector<Vec3f> vList(velocities);

		// Now populate the velocity attribute according to these values
		populateAttribute(pointDataGrid->tree(), pointIndexGrid->tree(), "velocity", vList);
	}

	void VDBParticleGrid::addParticle(const ponos::Point3& p, const ponos::vec3& v) {
		positions.emplace_back(Vec3f(p.x, p.y, p.z));
		velocities.emplace_back(Vec3f(v.x, v.y, v.z));
	}

	void VDBParticleGrid::addParticle(const ponos::ivec3& c, const ponos::Point3& p, const ponos::vec3& v) {

	}

	void VDBParticleGrid::iterateNeighbours(ponos::BBox bbox, std::function<void(const Particle& p)> f) {

	}

	void VDBParticleGrid::iterateCellNeighbours(const ponos::ivec3& c, const ponos::ivec3& d, std::function<void(const size_t& id)> f) {
		ponos::ivec3 pMin, pMax;
		pMax = c + d;
		ponos::ivec3 coord;
		int& x = coord[0], &y = coord[0], &z = coord[0];
		for(x = c[0] - d[0]; x < c[0] + d[0]; x++)
			for(y = c[1] - d[1]; y < c[1] + d[1]; y++)
				for(z = c[2] - d[2]; z < c[2] + d[2]; z++) {
					if(coord >= pMax || coord < pMin)
						continue;
					// iterate cell particles
					iterateCell(coord, f);
				}
	}

	void VDBParticleGrid::iterateCell(const ponos::ivec3& c, const std::function<void(const size_t& id)>& f) {
		// Touch a leaf at the origin to create it and confirm just one leaf
		PointDataTree::LeafNodeType* leaf = pointDataGrid->tree().touchLeaf(openvdb::Coord(0, 0, 0));

		// Retrieve a read-only attribute handle for position
		AttributeHandle<openvdb::Int32>::Ptr attributeHandle =
			AttributeHandle<openvdb::Int32>::create(leaf->attributeArray("id"));

		// Create a co-ordinate to perform the look-up and voxel position in index space
		const openvdb::Coord ijk(c[0], c[1], c[2]);

		// Create an IndexIter for accessing the co-ordinate
		IndexIter indexIter = leaf->beginIndex(ijk);

		// Iterate over all the points in the voxel
		for (; indexIter; ++indexIter) {
			f(attributeHandle->get(*indexIter));
		}
	}

} // poseidon namespace
