#include "scene/triangle_mesh.h"

namespace aergia {

	TriangleMesh::TriangleMesh(const std::string &filename) {
		RawMesh *m = new RawMesh();
		loadOBJ(filename, m);
		mesh.reset(m);
	}

	TriangleMesh::TriangleMesh(const RawMesh *m) {
		mesh.reset(m);
	}

	void TriangleMesh::draw() const {}

} // aergia namespace
