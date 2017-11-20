#include "structures/vdb_grid.h"

namespace poseidon {

	VDBGrid::VDBGrid(const ponos::ivec3& d, const float& b, const float& s, const ponos::vec3& o) {
		dimensions = d;
		background = b;
		// create grid
		openvdb::initialize();
		grid = openvdb::FloatGrid::create(b);
		// world transform
		grid->setTransform(openvdb::math::Transform::createLinearTransform(s));
		toWorld = ponos::scale(s, s, s);
		toGrid = ponos::inverse(toWorld);
		grid->setGridClass(openvdb::GRID_LEVEL_SET);
	}

	void VDBGrid::set(const ponos::ivec3& i, const float& v) {
		typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
		openvdb::Coord ijk(i[0], i[1], i[2]);
		accessor.setValue(ijk, v);
	}

	float VDBGrid::operator()(const ponos::ivec3& i) const {
		typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
		openvdb::Coord ijk(i[0], i[1], i[2]);
		return accessor.getValue(ijk);
	}

	float VDBGrid::operator()(const int& i, const int&j, const int& k) const {
		typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
		openvdb::Coord ijk(i, j, k);
		return accessor.getValue(ijk);
	}

	float VDBGrid::operator()(const ponos::vec3& i) const {
		typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
		openvdb::Coord ijk(i[0], i[1], i[2]);
		return accessor.getValue(ijk);
	}

	float VDBGrid::operator()(const float& i, const float&j, const float& k) const {
		typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
		openvdb::Coord ijk(i, j, k);
		return accessor.getValue(ijk);
	}

} // poseidon namespace
