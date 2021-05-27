#include <ponos/numeric/numeric.h>
#include <ponos/numeric/level_set.h>

namespace ponos {

LevelSet::LevelSet(const size3 &d, const vec3 cellSize, const vec3 &offset)
    : CRegularGrid<float>(d, 0.f, cellSize, offset) {}
void LevelSet::setMesh(const RawMesh *mesh) {
  PONOS_UNUSED_VARIABLE(mesh);
  setAll(INFINITY);
}
LevelSet2D::LevelSet2D(const ponos::RawMesh *m, const ponos::Transform2 &t) {
  PONOS_UNUSED_VARIABLE(m);
  PONOS_UNUSED_VARIABLE(t);
}
LevelSet2D::LevelSet2D(const size2 &d, const vec2 cellSize,
                       const vec2 &offset) {
  PONOS_UNUSED_VARIABLE(d);
  PONOS_UNUSED_VARIABLE(cellSize);
  PONOS_UNUSED_VARIABLE(offset);
}
void LevelSet2D::setMesh(const RawMesh *mesh) { PONOS_UNUSED_VARIABLE(mesh); }
void LevelSet2D::merge(const LevelSet2D *ls) { PONOS_UNUSED_VARIABLE(ls); }
void LevelSet2D::copy(const LevelSet2D *ls) { PONOS_UNUSED_VARIABLE(ls); }

} // namespace ponos
