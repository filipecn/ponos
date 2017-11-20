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

	float VDBMacGrid::u(int i, int j, int k) {
		openvdb::VectorTree::ValueType v = grid->tree().getValue(openvdb::Coord(i, j , k));
		return v[0];
	}

	float VDBMacGrid::v(int i, int j, int k) {
		openvdb::VectorTree::ValueType v = grid->tree().getValue(openvdb::Coord(i, j , k));
		return v[1];
	}

	float VDBMacGrid::w(int i, int j, int k) {
		openvdb::VectorTree::ValueType v = grid->tree().getValue(openvdb::Coord(i, j , k));
		return v[2];
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

	void VDBMacGrid::setU(int i, int j, int k, float v) {
		openvdb::Coord coord(i, j , k);
		openvdb::VectorTree::ValueType vt = grid->tree().getValue(coord);
		vt[0] = v;
		grid->tree().setValue(coord, vt);
	}

	void VDBMacGrid::setV(int i, int j, int k, float v) {
		openvdb::Coord coord(i, j , k);
		openvdb::VectorTree::ValueType vt = grid->tree().getValue(coord);
		vt[1] = v;
		grid->tree().setValue(coord, vt);
	}

	void VDBMacGrid::setW(int i, int j, int k, float v) {
		openvdb::Coord coord(i, j , k);
		openvdb::VectorTree::ValueType vt = grid->tree().getValue(coord);
		vt[2] = v;
		grid->tree().setValue(coord, vt);
	}

	void VDBMacGrid::computeDivergence() {
		divGrid = openvdb::tools::divergence(*grid);
	}

	float VDBMacGrid::getDivergence(int i, int j, int k) {
		openvdb::FloatGrid::ConstAccessor accessor = divGrid->getConstAccessor();
		openvdb::Coord xyz(i, j, k);
  	return accessor.getValue(xyz);
	}

} // poseidon namespace
