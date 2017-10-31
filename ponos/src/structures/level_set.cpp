#include "structures/level_set.h"

#include "geometry/numeric.h"

namespace ponos {

LevelSet::LevelSet(const ivec3 &d, const vec3 cellSize, const vec3 &offset)
    : CRegularGrid<float>(d, 0.f, cellSize, offset) {}
void LevelSet::setMesh(const RawMesh *mesh) {
  UNUSED_VARIABLE(mesh);
  setAll(INFINITY);
}
LevelSet2D::LevelSet2D(const ponos::RawMesh *m, const ponos::Transform2D &t) {
  UNUSED_VARIABLE(m);
  UNUSED_VARIABLE(t);
}
LevelSet2D::LevelSet2D(const ivec2 &d, const vec2 cellSize,
                       const vec2 &offset) {
  UNUSED_VARIABLE(d);
  UNUSED_VARIABLE(cellSize);
  UNUSED_VARIABLE(offset);
}
void LevelSet2D::setMesh(const RawMesh *mesh) { UNUSED_VARIABLE(mesh); }
void LevelSet2D::merge(const LevelSet2D *ls) { UNUSED_VARIABLE(ls); }
void LevelSet2D::copy(const LevelSet2D *ls) { UNUSED_VARIABLE(ls); }

} // ponos namespace
