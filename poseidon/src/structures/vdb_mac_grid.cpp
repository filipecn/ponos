#include "structures/vdb_mac_grid.h"
#include <openvdb/tools/GridOperators.h>

namespace poseidon {

	VDBMacGrid::VDBMacGrid(const ponos::ivec3& d, const float& b, const float& s, const ponos::vec3& o) {
		dimensions = d;
		// create grid
		openvdb::initialize();
		grid = openvdb::VectorGrid::create();
		// world transform
		grid->setTransform(openvdb::math::Transform::createLinearTransform(s));
		grid->setGridClass(openvdb::GRID_STAGGERED);
	}

	ponos::Point3 VDBMacGrid::getWorldPositionX(int i, int j, int k) const {
		return toWorld(ponos::Point3(i - 0.5f, j, k));
	}

	ponos::Point3 VDBMacGrid::getWorldPositionY(int i, int j, int k) const {
		return toWorld(ponos::Point3(i, j - 0.5f, k));
	}

	ponos::Point3 VDBMacGrid::getWorldPositionZ(int i, int j, int k) const {
		return toWorld(ponos::Point3(i, j, k - 0.5f));
	}

	void VDBMacGrid::set(int i, int j, int k, ponos::vec3 v) {

	}

	void VDBMacGrid::setX(int i, int j, int k, float v) {}

	void VDBMacGrid::setY(int i, int j, int k, float v) {}

	void VDBMacGrid::setZ(int i, int j, int k, float v) {}

	void VDBMacGrid::computeDivergence() {
		divGrid = openvdb::tools::divergence(*grid);
		// openvdb::FloatGrid::ConstAccessor accessor = divGrid->getConstAccessor();
	}

} // poseidon namespace
