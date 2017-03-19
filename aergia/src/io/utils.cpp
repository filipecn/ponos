/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#include "io/utils.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <tiny_obj_loader.h>

#include <cstring>

namespace aergia {

void loadOBJ(const std::string &filename, ponos::RawMesh *mesh) {
  if (!mesh)
    return;
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;
  bool r =
      tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
  if (!r)
    return;
  mesh->vertices = std::vector<float>(attrib.vertices);
  mesh->normals = std::vector<float>(attrib.normals);
  mesh->texcoords = std::vector<float>(attrib.texcoords);
  mesh->indices.resize(shapes[0].mesh.indices.size());
  memcpy(&mesh->indices[0], &shapes[0].mesh.indices[0],
         shapes[0].mesh.indices.size() * sizeof(tinyobj::index_t));
  mesh->vertexDescriptor.count =
      mesh->vertices.size() / mesh->vertexDescriptor.elementSize;
  mesh->meshDescriptor.elementSize = 3;
  mesh->meshDescriptor.count =
      mesh->indices.size() / mesh->meshDescriptor.elementSize;
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
                          tinyobj::index_t idx =
  shapes[s].mesh.indices[index_offset + v];
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
