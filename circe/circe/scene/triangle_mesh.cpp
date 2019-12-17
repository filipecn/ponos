#include <circe/scene/triangle_mesh.h>

#include <iostream>

namespace circe {

TriangleMesh::TriangleMesh(const std::string &filename)
    : SceneMeshObject(filename) {}

TriangleMesh::TriangleMesh(const ponos::RawMesh *m) : SceneMeshObject(m) {}

void TriangleMesh::draw(const CameraInterface *camera,
                        ponos::Transform transform) {}

} // namespace circe
