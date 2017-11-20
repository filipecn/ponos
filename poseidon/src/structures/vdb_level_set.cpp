#include "structures/vdb_level_set.h"

#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToSpheres.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/NullInterrupter.h>

namespace poseidon {

VDBLevelSet::VDBLevelSet(const ponos::RawMesh *m, ponos::Transform t) {
  openvdb::math::Transform::Ptr transform =
      openvdb::math::Transform::createLinearTransform(.25f);
  // copy vertices
  std::vector<openvdb::Vec3s> points;
  size_t vertexCount = m->vertices.size() / 3;
  for (size_t i = 0; i < vertexCount; i++) {
    ponos::Point3 p = t(ponos::Point3(&m->vertices[i * 3]));
    points.push_back(transform->worldToIndex(openvdb::Vec3s(p.x, p.y, p.z)));
  }
  // copy faces
  std::vector<openvdb::Vec4I> faces;
  size_t elementCount =
      m->indices.size() / m->interleavedDescriptor.elementSize;
  std::cout << m->verticesIndices.size() << std::endl;
  for (size_t i = 0; i < elementCount; i++) {
    ponos::uivec4 face(
        m->interleavedDescriptor.elementSize,
        &m->verticesIndices[i * m->interleavedDescriptor.elementSize]);
    openvdb::Vec4I vdbpoly(face[0], face[1], face[2], face[3]);
    if (m->interleavedDescriptor.elementSize == 3)
      vdbpoly[3] = openvdb::util::INVALID_IDX;
    faces.push_back(vdbpoly);
  }

  grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*transform, points,
                                                            faces);
}

void VDBLevelSet::set(const ponos::ivec3 &ijk, const float &v) {
  // m_setCellLock.lock();
  openvdb::Coord coord = openvdb::Coord(ijk[0], ijk[1], ijk[2]);
  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
  accessor.setValue(coord, v);
  // m_setCellLock.unlock();
}

float VDBLevelSet::operator()(const ponos::ivec3 &ijk) const {
  return (*this)(ijk[0], ijk[1], ijk[2]);
}

float VDBLevelSet::operator()(const int &i, const int &j, const int &k) const {
  openvdb::Coord coord = openvdb::Coord(i, j, k);
  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
  return accessor.getValue(coord);
}

float VDBLevelSet::operator()(const ponos::vec3 &xyz) const {
  return (*this)(xyz.x, xyz.y, xyz.z);
}

float VDBLevelSet::operator()(const float &x, const float &y,
                              const float &z) const {
  float value;
  // m_getInterpolatedCellLock.lock();
  openvdb::Vec3f p(x, y, z);
  openvdb::tools::GridSampler<openvdb::FloatTree, openvdb::tools::BoxSampler>
      interpolator(grid->constTree(), grid->transform());
  value = interpolator.wsSample(p);
  // m_getInterpolatedCellLock.unlock();
  return value;
}

void VDBLevelSet::merge(const VDBLevelSet *ls) {
  openvdb::FloatGrid::Ptr objectSDF = ls->getVDBGrid()->deepCopy();
  openvdb::tools::csgUnion(*grid, *objectSDF);
  objectSDF->clear();
}

void VDBLevelSet::copy(const VDBLevelSet *ls) {
  grid = ls->getVDBGrid()->deepCopy();
}

const openvdb::FloatGrid::Ptr &VDBLevelSet::getVDBGrid() const { return grid; }

} // poseidon namespace
