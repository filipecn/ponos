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

#include "structures/raw_mesh.h"

namespace ponos {

void RawMesh::addVertex(std::initializer_list<float> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    vertices.emplace_back(*it);
}

void RawMesh::addFace(std::initializer_list<IndexData> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    indices.emplace_back(*it);
}

void RawMesh::apply(const Transform &t) {
  size_t nv = vertices.size() / vertexDescriptor.elementSize;
  for (size_t i = 0; i < nv; i++)
    t.applyToPoint(&vertices[i * vertexDescriptor.elementSize],
                   &vertices[i * vertexDescriptor.elementSize]);
}

void RawMesh::splitIndexData() {
  if (verticesIndices.size())
    return;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    verticesIndices.emplace_back(indices[i].vertexIndex);
    normalsIndices.emplace_back(indices[i].normalIndex);
    texcoordsIndices.emplace_back(indices[i].texcoordIndex);
  }
}

void RawMesh::computeBBox() {
  bbox = BBox();
  for (size_t i = 0; i < vertexDescriptor.count; i++) {
    ponos::Point3 p;
    for (size_t d = 0; d < vertexDescriptor.elementSize; d++)
      p[d] = vertices[i * vertexDescriptor.elementSize + d];
    bbox = make_union(bbox, p);
  }
}

Point3 RawMesh::vertexElement(size_t e, size_t v) const {
  return Point3(
      vertices[indices[e * meshDescriptor.elementSize + v].vertexIndex *
                   vertexDescriptor.elementSize +
               0],
      vertices[indices[e * meshDescriptor.elementSize + v].vertexIndex *
                   vertexDescriptor.elementSize +
               1],
      vertices[indices[e * meshDescriptor.elementSize + v].vertexIndex *
                   vertexDescriptor.elementSize +
               2]);
}

BBox RawMesh::elementBBox(size_t i) const {
  BBox b;
  for (size_t v = 0; v < meshDescriptor.elementSize; v++)
    b = make_union(
        b,
        Point3(
            vertices[indices[i * meshDescriptor.elementSize + v].vertexIndex *
                         vertexDescriptor.elementSize +
                     0],
            vertices[indices[i * meshDescriptor.elementSize + v].vertexIndex *
                         vertexDescriptor.elementSize +
                     1],
            vertices[indices[i * meshDescriptor.elementSize + v].vertexIndex *
                         vertexDescriptor.elementSize +
                     2]));
  return b;
}

void RawMesh::buildInterleavedData() {
  interleavedData.clear();
  for (size_t i = 0; i < vertexDescriptor.count; i++) {
    for (size_t k = 0; k < vertexDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          vertices[i * vertexDescriptor.elementSize + k]);
    for (size_t k = 0; k < normalDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          normals[i * normalDescriptor.elementSize + k]);
    for (size_t k = 0; k < texcoordDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          texcoords[i * texcoordDescriptor.elementSize + k]);
  }
  interleavedDescriptor.elementSize = vertexDescriptor.elementSize +
                                      normalDescriptor.elementSize +
                                      texcoordDescriptor.elementSize;
  interleavedDescriptor.count = vertexDescriptor.count;
  ASSERT(interleavedData.size() ==
         vertexDescriptor.count * vertexDescriptor.elementSize +
             normalDescriptor.count * normalDescriptor.elementSize +
             texcoordDescriptor.count * texcoordDescriptor.elementSize);
}

void RawMesh::orientFaces(bool ccw) {
  for (size_t e = 0; e < meshDescriptor.count; e++) {
    bool flip = false;
    for (int i = 0; i < meshDescriptor.elementSize; i++) {
      ponos::Point3 a3 = vertexElement(e, i);
      ponos::Point3 b3 = vertexElement(e, i + 1 % meshDescriptor.elementSize);
      ponos::Point3 c3 = vertexElement(e, i + 2 % meshDescriptor.elementSize);
      ponos::Point2 a(a3.x, a3.y);
      ponos::Point2 b(b3.x, b3.y);
      ponos::Point2 c(c3.x, c3.y);
      if (ccw && ponos::cross(b - a, c - a) < 0.f)
        flip = true;
    }
    std::cout << "face " << e << " " << flip << std::endl;
    if (flip) {
      IndexData tmp = indices[e * meshDescriptor.elementSize + 0];
      indices[e * meshDescriptor.elementSize + 0] =
          indices[e * meshDescriptor.elementSize + 1];
      indices[e * meshDescriptor.elementSize + 1] = tmp;
    }
  }
}

} // aergia namespace
