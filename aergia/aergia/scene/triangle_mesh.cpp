#include <aergia/scene/triangle_mesh.h>

#include <iostream>

namespace aergia {

TriangleMesh::TriangleMesh(const std::string &filename) : SceneMesh(filename) {}

TriangleMesh::TriangleMesh(ponos::RawMesh *m) : SceneMesh(m) {}

void TriangleMesh::draw() const {}

} // aergia namespace
