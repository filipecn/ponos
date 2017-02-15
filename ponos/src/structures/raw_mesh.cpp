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
		for(size_t i = 0; i < vertexCount; i++) {
			ponos::Point3 p;
			for(size_t d = 0; d < dimensions; d++)
				p[d] = vertices[i * dimensions + d];
			bbox = make_union(bbox, p);
		}
	}

	Point3 RawMesh::vertexElement(size_t e, size_t v) const {
		return Point3(
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 0],
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 1],
				vertices[indices[e * elementSize + v].vertexIndex * elementSize + 2]);
	}

	BBox RawMesh::elementBBox(size_t i) const {
		BBox b;
		for(size_t v = 0; v < elementSize; v++)
			b = make_union(b, Point3(
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 0],
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 1],
						vertices[indices[i * elementSize + v].vertexIndex * elementSize + 2]));
		return b;
	}

} // aergia namespace
