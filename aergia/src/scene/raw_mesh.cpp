#include "scene/raw_mesh.h"

namespace aergia {

	void RawMesh::apply(const ponos::Transform &t) {
		size_t nv = vertices.size() / 3;
		for(size_t i = 0; i < nv; i++)
			t.applyToPoint(&vertices[i * 3], &vertices[i * 3]);
	}

	void RawMesh::splitIndexData() {
		if(verticesIndices.size())
			return;
		size_t size = indices.size();
		for(size_t i = 0; i < size; i++) {
			verticesIndices.emplace_back(indices[i].vertexIndex);
			normalsIndices.emplace_back(indices[i].normalIndex);
			texcoordsIndices.emplace_back(indices[i].texcoordIndex);
		}
	}

	void RawMesh::computeBBox() {
		bbox = ponos::BBox();
		size_t n = vertices.size() / 3;
		for(size_t i = 0; i < n; i++)
			bbox = ponos::make_union(bbox, ponos::Point3(
						vertices[i * 3 + 0],
						vertices[i * 3 + 1],
						vertices[i * 3 + 2]));
	}

} // aergia namespace
