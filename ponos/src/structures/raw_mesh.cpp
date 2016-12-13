#include "structures/raw_mesh.h"

namespace ponos {

	void RawMesh::apply(const Transform &t) {
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
		bbox = BBox();
		size_t n = vertices.size() / 3;
		for(size_t i = 0; i < n; i++)
			bbox = make_union(bbox, Point3(
						vertices[i * 3 + 0],
						vertices[i * 3 + 1],
						vertices[i * 3 + 2]));
	}

	Point3 RawMesh::vertexElement(size_t e, size_t v) const {
		return Point3(
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 0],
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 1],
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 2]);
	}

	BBox RawMesh::elementBBox(size_t i) const {
		BBox b;
		for(int v = 0; v < elementSize; v++)
			b = make_union(b, Point3(
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 0],
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 1],
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 2]));
		return b;
	}

} // aergia namespace
