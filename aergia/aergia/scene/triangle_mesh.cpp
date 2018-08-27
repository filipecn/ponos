#include <aergia/scene/triangle_mesh.h>

#include <iostream>

namespace aergia {

TriangleMesh::TriangleMesh(const std::string &filename) : SceneMeshObject(filename) {}

TriangleMesh::TriangleMesh(ponos::RawMesh *m) : SceneMeshObject(m) {}

void TriangleMesh::draw(const CameraInterface *camera, ponos::Transform transform) {}

} // aergia namespace
