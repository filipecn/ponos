#include "structures/vdb_mac_grid.h"

namespace poseidon {

	VDBMacGrid::VDBMacGrid(const ponos::ivec3& d, const float& b, const float& s, const ponos::vec3& o) {
		dimensions = d;
		// create grid
		openvdb::initialize();
		v = openvdb::VectorGrid::create();
		// world transform
		v->setTransform(openvdb::math::Transform::createLinearTransform(s));
		v->setGridClass(openvdb::GRID_STAGGERED);
	}

} // poseidon namespace
