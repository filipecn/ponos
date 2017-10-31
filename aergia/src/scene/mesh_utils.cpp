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

#include "scene/mesh_utils.h"

namespace aergia {

ponos::RawMesh *create_grid_mesh(const ponos::ivec3 &d, float s,
                                 const ponos::vec3 &o) {
  ponos::RawMesh *m = new ponos::RawMesh();
  ponos::ivec2 ij;
  // XY
  FOR_INDICES0_2D(d.xy(0, 1), ij) {
    ponos::Point3 p(ij[0], ij[1], 0.f);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  FOR_INDICES0_2D(d.xy(0, 1), ij) {
    ponos::Point3 p(ij[0], ij[1], d[2] - 1);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  // YZ
  FOR_INDICES0_2D(d.xy(1, 2), ij) {
    ponos::Point3 p(0.f, ij[0], ij[1]);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  FOR_INDICES0_2D(d.xy(1, 2), ij) {
    ponos::Point3 p(d[0] - 1, ij[0], ij[1]);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  // XZ
  FOR_INDICES0_2D(d.xy(0, 2), ij) {
    ponos::Point3 p(ij[0], 0.f, ij[1]);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  FOR_INDICES0_2D(d.xy(0, 2), ij) {
    ponos::Point3 p(ij[0], d[1] - 1, ij[1]);
    p = p * s + o;
    m->vertices.emplace_back(p.x);
    m->vertices.emplace_back(p.y);
    m->vertices.emplace_back(p.z);
  }
  m->vertexDescriptor.count = m->vertices.size() / 3;
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

ponos::RawMesh *create_wireframe_mesh(const ponos::RawMesh *m) {
  ponos::RawMesh *mesh = new ponos::RawMesh();
  mesh->vertexDescriptor.count = m->vertexDescriptor.count;
  mesh->vertices = std::vector<float>(m->vertices);
  size_t nelements = m->indices.size() / m->meshDescriptor.elementSize;
  for (size_t i = 0; i < nelements; i++)
    for (size_t j = 0; j < m->meshDescriptor.elementSize; j++) {
      mesh->verticesIndices.emplace_back(
          m->indices[i * m->meshDescriptor.elementSize + j].vertexIndex);
      mesh->verticesIndices.emplace_back(
          m->indices[i * m->meshDescriptor.elementSize +
                     (j + 1) % m->meshDescriptor.elementSize].vertexIndex);
    }
  return mesh;
}

} // aergia namespace
