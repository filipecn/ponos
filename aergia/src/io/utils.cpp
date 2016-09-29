#include "io/utils.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <tiny_obj_loader.h>

#include "scene/raw_mesh.h"

#include <cstring>

namespace aergia {

	void loadOBJ(const std::string& filename, RawMesh *mesh) {
		if(!mesh)
			return;
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;
		bool r = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
		if(!r)
			return;
		mesh->vertices = std::vector<float>(attrib.vertices);
		mesh->normals = std::vector<float>(attrib.normals);
		mesh->texcoords = std::vector<float>(attrib.texcoords);
		mesh->indices.resize(shapes[0].mesh.indices.size());
		memcpy(&mesh->indices[0], &shapes[0].mesh.indices[0],
				shapes[0].mesh.indices.size() * sizeof(tinyobj::index_t));
		mesh->vertexCount = mesh->vertices.size() / 3;
		mesh->elementSize = 3;
		mesh->computeBBox();
		mesh->splitIndexData();
		/* tiny obj use
		// Loop over shapes
		for (size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces(polygon)
			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				int fv = shapes[s].mesh.num_face_vertices[f];
				// Loop over vertices in the face.
				for (size_t v = 0; v < fv; v++) {
					// access to vertex
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					float vx = attrib.vertices[3*idx.vertex_index+0];
					float vy = attrib.vertices[3*idx.vertex_index+1];
					float vz = attrib.vertices[3*idx.vertex_index+2];
					float nx = attrib.normals[3*idx.normal_index+0];
					float ny = attrib.normals[3*idx.normal_index+1];
					float nz = attrib.normals[3*idx.normal_index+2];
					float tx = attrib.texcoords[2*idx.texcoord_index+0];
					float ty = attrib.texcoords[2*idx.texcoord_index+1];
				}
				index_offset += fv;

				// per-face material
				shapes[s].mesh.material_ids[f];
			}
		}*/
	}

} // aergia namespace
