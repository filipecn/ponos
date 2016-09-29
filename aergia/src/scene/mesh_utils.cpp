#include "scene/mesh_utils.h"

namespace aergia {

	RawMesh *create_grid_mesh(const ponos::ivec3 &d, float s, const ponos::vec3 &o) {
		RawMesh *m = new RawMesh();
		ponos::ivec2 ij;
		// XY
		FOR_INDICES0_2D(d.xy(0, 1), ij) {
			ponos::Point3 p(ij[0], ij[1], 0.f);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		FOR_INDICES0_2D(d.xy(0, 1), ij) {
			ponos::Point3 p(ij[0], ij[1], d[2] - 1);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		// YZ
		FOR_INDICES0_2D(d.xy(1, 2), ij) {
			ponos::Point3 p(0.f, ij[0], ij[1]);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		FOR_INDICES0_2D(d.xy(1, 2), ij) {
			ponos::Point3 p(d[0] - 1, ij[0], ij[1]);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		// XZ
		FOR_INDICES0_2D(d.xy(0, 2), ij) {
			ponos::Point3 p(ij[0], 0.f,  ij[1]);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		FOR_INDICES0_2D(d.xy(0, 2), ij) {
			ponos::Point3 p(ij[0], d[1] - 1, ij[1]);
			p = p * s + o;
			m->vertices.emplace_back(p.x);m->vertices.emplace_back(p.y);m->vertices.emplace_back(p.z);
		}
		m->vertexCount = m->vertices.size () / 3;
		int xy = d[0] * d[1];
		// create indices for xy planes
		FOR_INDICES0_2D(d.xy(0, 1), ij) {
			m->verticesIndices.emplace_back(ij[0] * d[0] + ij[1]);
			m->verticesIndices.emplace_back(xy + ij[0] * d[0] + ij[1]);
		}
		int acc = xy * 2;
		int yz = d[1] * d[2];
		// create indices for yz planes
		FOR_INDICES0_2D(d.xy(1, 2), ij) {
			m->verticesIndices.emplace_back(acc + ij[0] * d[1] + ij[1]);
			m->verticesIndices.emplace_back(acc + yz + ij[0] * d[1] + ij[1]);
		}
		acc += yz * 2;
		int xz = d[0] * d[2];
		// create indices for xz planes
		FOR_INDICES0_2D(d.xy(1, 2), ij) {
			m->verticesIndices.emplace_back(acc + ij[0] * d[1] + ij[1]);
			m->verticesIndices.emplace_back(acc + xz + ij[0] * d[1] + ij[1]);
		}
		return m;
	}

	RawMesh *create_wireframe_mesh(const RawMesh *m) {
		RawMesh *mesh = new RawMesh();
		mesh->vertexCount = m->vertexCount;
		mesh->vertices = std::vector<float>(m->vertices);
		size_t nelements = m->indices.size() / m->elementSize;
		for(size_t i = 0; i < nelements; i++)
			for(size_t j = 0; j < m->elementSize; j++) {
				mesh->verticesIndices.emplace_back(m->indices[i * m->elementSize + j].vertexIndex);
				mesh->verticesIndices.emplace_back(m->indices[i * m->elementSize + (j + 1) % m->elementSize].vertexIndex);
			}
		return mesh;
	}

} // aergia namespace
