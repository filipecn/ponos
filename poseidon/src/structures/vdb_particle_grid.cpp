#include "structures/vdb_particle_grid.h"

#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/tools/PointCount.h>

using namespace openvdb;
using namespace openvdb::tools;

namespace poseidon {

	VDBParticleGrid::VDBParticleGrid(const ponos::ivec3& d, const float& s, const ponos::vec3& o)
		: dimensions(d), scale(s), updated(false) {
			// initialize libraries
			openvdb::initialize();
			openvdb::points::initialize();
			// Introduce a typedef for our position attribute (note no compression codec here)
			typedef TypedAttributeArray<Vec3f> PositionAttribute;
			typedef TypedAttributeArray<Vec3f> VelocityAttribute;
			typedef TypedAttributeArray<Int32> IdAttribute;
			// Create a list of names and attribute types (in this case, just position)
			AttributeSet::Descriptor::NameAndTypeVec attributes;
			attributes.push_back(AttributeSet::Descriptor::NameAndType("P", PositionAttribute::attributeType()));
			attributes.push_back(AttributeSet::Descriptor::NameAndType("V", VelocityAttribute::attributeType()));
			attributes.push_back(AttributeSet::Descriptor::NameAndType("I", IdAttribute::attributeType()));
			// Create an AttributeSet Descriptor for this list
			descriptor = AttributeSet::Descriptor::create(attributes);
			leaf = nullptr;

			transform = openvdb::math::Transform::createLinearTransform(scale);
			toWorld = ponos::scale(scale, scale, scale);
			toGrid = ponos::inverse(toWorld);
		}

	void VDBParticleGrid::init() {
		if(updated)
			return;
		// Create a PointPartitioner-compatible point list using the std::vector wrapper provided
		const PointAttributeVector<openvdb::Vec3f> pointList(positions);
		pointIndexGrid = createPointIndexGrid<PointIndexGrid>(pointList, *transform);
		// Create the PointDataGrid, position attribute is mandatory
		pointDataGrid = createPointDataGrid<PointDataGrid>(*pointIndexGrid, pointList, TypedAttributeArray<openvdb::Vec3f>::attributeType(), *transform);

		// Add a new velocity attribute
		AttributeSet::Util::NameAndType velocityAttribute("V", TypedAttributeArray<openvdb::Vec3f>::attributeType());
		appendAttribute(pointDataGrid->tree(), velocityAttribute);
		const PointAttributeVector<Vec3f> vList(velocities);
		populateAttribute(pointDataGrid->tree(), pointIndexGrid->tree(), "V", vList);

		// Add a new id attribute
		AttributeSet::Util::NameAndType idAttribute("id", TypedAttributeArray<openvdb::Int32>::attributeType());
		appendAttribute(pointDataGrid->tree(), idAttribute);
		const PointAttributeVector<Int32> idList(ids);
		populateAttribute(pointDataGrid->tree(), pointIndexGrid->tree(), "id", idList);

		// Add a new id attribute
		AttributeSet::Util::NameAndType densityAttribute("D", TypedAttributeArray<float>::attributeType());
		appendAttribute(pointDataGrid->tree(), densityAttribute);
		const PointAttributeVector<float> densityList(densities);
		populateAttribute(pointDataGrid->tree(), pointIndexGrid->tree(), "D", densityList);

		leaf = pointDataGrid->tree().touchLeaf(openvdb::Coord(0, 0, 0));

		updated = true;
	}

	void VDBParticleGrid::addParticle(const ponos::Point3& p, const ponos::vec3& v) {
		ponos::ivec3 i = worldToIndex(p);
		if(!(i >= ponos::ivec3()) || !(i < dimensions))
			return;
		ids.emplace_back(positions.size());
		positions.emplace_back(Vec3f(p.x, p.y, p.z));
		velocities.emplace_back(Vec3f(v.x, v.y, v.z));
		densities.emplace_back(0.f);
		updated = false;
	}

	void VDBParticleGrid::addParticle(const ponos::ivec3& c, const ponos::Point3& p, const ponos::vec3& v) {
		ponos::Point3 tp = (p + ponos::vec3(c[0], c[1], c[2])) * scale;
		addParticle(tp, v);
	}

	void VDBParticleGrid::setParticle(int id, const ponos::Point3& p, const ponos::vec3& v) {
		/*AttributeWriteHandle<Vec3f>::Ptr attributeWriteHandle =
			AttributeWriteHandle<Vec3f>::create(leaf->attributeArray("P"));
		attributeWriteHandle->set(id, openvdb::Vec3f(p.x, p.y, p.z));
		attributeWriteHandle =
			AttributeWriteHandle<Vec3f>::create(leaf->attributeArray("V"));
		attributeWriteHandle->set(id, Vec3f(v.x, v.y, v.z));*/
		positions[id] = Vec3f(p.x, p.y, p.z);
		velocities[id] = Vec3f(v.x, v.y, v.z);
		updated = false;
	}

	int VDBParticleGrid::particleCount(const ponos::ivec3& ijk) {
		if(!updated)
			init();
		const openvdb::Coord _ijk(ijk[0], ijk[1], ijk[2]);
		IndexIter indexIter = leaf->beginIndex(_ijk);
		return iterCount(indexIter);
	}

	void VDBParticleGrid::iterateNeighbours(ponos::BBox bbox, std::function<void(const Particle& p)> f) {
		if(!updated)
			init();
	}

	void VDBParticleGrid::iterateCellNeighbours(const ponos::ivec3& c, const ponos::ivec3& d, std::function<void(const size_t& id)> f) {
		if(!updated)
			init();
		ponos::ivec3 pMin(0, 0, 0);
		ponos::ivec3 coord;
		int& x = coord[0], &y = coord[1], &z = coord[2];
		for(x = c[0] - d[0]; x <= c[0] + d[0]; x++)
			for(y = c[1] - d[1]; y <= c[1] + d[1]; y++)
				for(z = c[2] - d[2]; z <= c[2] + d[2]; z++) {
					if(!(coord < dimensions) || !(coord >= pMin))
						continue;
					// iterate cell particles
					iterateCell(coord, f);
				}
	}

	void VDBParticleGrid::iterateCell(const ponos::ivec3& c, const std::function<void(const size_t& id)>& f) {
		if(!updated)
			init();
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

	int VDBParticleGrid::particleCount() {
		if(!updated)
			init();
		if(pointDataGrid == nullptr)
			return 0;
		return pointCount(pointDataGrid->tree());
	}

	ponos::ivec3 VDBParticleGrid::worldToIndex(const ponos::Point3& wp) {
		ponos::Point3 gp = toGrid(wp) + ponos::vec3(0.5f, 0.5f, 0.5f);
		return ponos::ivec3(gp.x, gp.y, gp.z);
	}

	ponos::ivec3 VDBParticleGrid::gridToIndex(const ponos::Point3& gp) {
		ponos::Point3 p = gp + ponos::vec3(0.5f, 0.5f, 0.5f);
		return ponos::ivec3(p.x, p.y, p.z);
	}

	ponos::Point3 VDBParticleGrid::worldToVoxel(const ponos::Point3& wp) {
		ponos::Point3 gp = toGrid(wp);
		ponos::ivec3 ip = gridToIndex(gp);
		return ponos::Point3(gp - ponos::vec3(ip[0], ip[1], ip[2]));
	}

	ponos::Point3 VDBParticleGrid::gridToVoxel(const ponos::Point3& gp) {
		ponos::ivec3 ip = gridToIndex(gp);
		return ponos::Point3(gp - ponos::vec3(ip[0], ip[1], ip[2]));
	}

	float VDBParticleGrid::computeDensity(float d, float md) {
		if(!updated)
			init();
		float maxd = dimensions.max();
		ponos::parallel_for(0, positions.size(), [this, maxd, d, md](size_t f, size_t l) {
				for(size_t id = f; id <= l; id++) {
				ponos::ivec3 cell = worldToIndex(ponos::Point3(positions[id].asPointer()));
				float sum = 0.f;
				iterateCellNeighbours(cell, ponos::ivec3(1, 1, 1), [this, &sum, d, id, maxd](const size_t& i) {
						float distance = ponos::distance2(
									ponos::Point3(positions[id].asPointer()),
									ponos::Point3(positions[i].asPointer()));
						float weight = particleMass * ponos::smooth(distance, 4.f * d / maxd);
						sum += weight;
						});
				densities[id] = sum / md;
				}
				});
		size_t _md = ponos::parallel_max(0, densities.size(), &densities[0]);
		return densities[_md];
	}
} // poseidon namespace
