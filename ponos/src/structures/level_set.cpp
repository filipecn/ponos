#include "structures/level_set.h"

#include "geometry/numeric.h"

namespace ponos {

	LevelSet::LevelSet(const ivec3& d,
			const vec3 cellSize,
			const vec3& offset)
		: CRegularGrid<float>(d, 0.f, cellSize, offset) {
		}

	void LevelSet::setMesh(const RawMesh* mesh) {
		setAll(INFINITY);
	}

	LevelSet2D::LevelSet2D(const ponos::RawMesh* m, const ponos::Transform2D& t) {}

	LevelSet2D::LevelSet2D(const ivec2& d, const vec2 cellSize, const vec2& offset) {}
	void LevelSet2D::setMesh(const RawMesh* mesh) {}


	void LevelSet2D::merge(const LevelSet2D *ls) {}
	void LevelSet2D::copy(const LevelSet2D *ls) {}

} // ponos namespace
