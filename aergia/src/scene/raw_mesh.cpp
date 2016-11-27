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

	ponos::Point3 RawMesh::vertexElement(size_t e, size_t v) const {
		return ponos::Point3(
				vertices[indices[e + v].vertexIndex * elementSize + 0],
				vertices[indices[e + v].vertexIndex * elementSize + 1],
				vertices[indices[e + v].vertexIndex * elementSize + 2]);
	}

	ponos::BBox RawMesh::elementBBox(size_t i) const {
		ponos::BBox b;
		for(int v = 0; v < elementSize; v++)
			b = ponos::make_union(b, ponos::Point3(
						vertices[indices[i + v].vertexIndex * elementSize + 0],
						vertices[indices[i + v].vertexIndex * elementSize + 1],
						vertices[indices[i + v].vertexIndex * elementSize + 2]));
		return b;
	}

} // aergia namespace
