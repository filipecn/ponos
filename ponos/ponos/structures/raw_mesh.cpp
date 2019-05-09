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

#include <ponos/structures/raw_mesh.h>

namespace ponos {

void RawMesh::addPosition(std::initializer_list<float> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    positions.emplace_back(*it);
}

void RawMesh::addNormal(std::initializer_list<float> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    normals.emplace_back(*it);
}

void RawMesh::addUV(std::initializer_list<float> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    texcoords.emplace_back(*it);
}

void RawMesh::addFace(std::initializer_list<IndexData> l) {
  for (auto it = l.begin(); it != l.end(); it++)
    indices.emplace_back(*it);
}

void RawMesh::addFace(std::initializer_list<int> l) {
  for (auto it = l.begin(); it != l.end(); it++) {
    IndexData id;
    id.normalIndex = id.texcoordIndex = 0;
    id.positionIndex = *it;
    indices.emplace_back(id);
  }
}

void RawMesh::apply(const Transform &t) {
  for (size_t i = 0; i < positionDescriptor.count; i++)
    t.applyToPoint(&positions[i * positionDescriptor.elementSize],
                   &positions[i * positionDescriptor.elementSize],
                   positionDescriptor.elementSize);
  if (interleavedData.size())
    buildInterleavedData();
}

void RawMesh::splitIndexData() {
  if (positionsIndices.size())
    return;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    positionsIndices.emplace_back(indices[i].positionIndex);
    normalsIndices.emplace_back(indices[i].normalIndex);
    texcoordsIndices.emplace_back(indices[i].texcoordIndex);
  }
}

void RawMesh::computeBBox() {
  bbox = bbox3();
  for (size_t i = 0; i < positionDescriptor.count; i++) {
    ponos::point3 p;
    for (size_t d = 0; d < positionDescriptor.elementSize; d++)
      p[d] = positions[i * positionDescriptor.elementSize + d];
    bbox = make_union(bbox, p);
  }
}

point3 RawMesh::positionElement(size_t e, size_t v) const {
  return point3(
      positions[indices[e * meshDescriptor.elementSize + v].positionIndex *
                    positionDescriptor.elementSize +
                0],
      positions[indices[e * meshDescriptor.elementSize + v].positionIndex *
                    positionDescriptor.elementSize +
                1],
      ((positionDescriptor.elementSize == 3)
           ? positions[indices[e * meshDescriptor.elementSize + v]
                               .positionIndex *
                           positionDescriptor.elementSize +
                       2]
           : 0.f));
}

bbox3 RawMesh::elementBBox(size_t i) const {
  bbox3 b;
  for (size_t v = 0; v < meshDescriptor.elementSize; v++)
    b = make_union(b,
                   point3(positions[indices[i * meshDescriptor.elementSize + v]
                                            .positionIndex *
                                        positionDescriptor.elementSize +
                                    0],
                          positions[indices[i * meshDescriptor.elementSize + v]
                                            .positionIndex *
                                        positionDescriptor.elementSize +
                                    1],
                          positions[indices[i * meshDescriptor.elementSize + v]
                                            .positionIndex *
                                        positionDescriptor.elementSize +
                                    2]));
  return b;
}

void RawMesh::buildInterleavedData() {
  interleavedData.clear();
  for (size_t i = 0; i < positionDescriptor.count; i++) {
    for (size_t k = 0; k < positionDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          positions[i * positionDescriptor.elementSize + k]);
    for (size_t k = 0; k < normalDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          normals[i * normalDescriptor.elementSize + k]);
    for (size_t k = 0; k < texcoordDescriptor.elementSize; k++)
      interleavedData.emplace_back(
          texcoords[i * texcoordDescriptor.elementSize + k]);
  }
  interleavedDescriptor.elementSize = positionDescriptor.elementSize +
                                      normalDescriptor.elementSize +
                                      texcoordDescriptor.elementSize;
  interleavedDescriptor.count = positionDescriptor.count;
  ASSERT_EQ(interleavedData.size(),
            positionDescriptor.count * positionDescriptor.elementSize +
                normalDescriptor.count * normalDescriptor.elementSize +
                texcoordDescriptor.count * texcoordDescriptor.elementSize);
}

void RawMesh::orientFaces(bool ccw) {
  for (size_t e = 0; e < meshDescriptor.count; e++) {
    bool flip = false;
    for (size_t i = 0; i < meshDescriptor.elementSize; i++) {
      ponos::point3 a3 = positionElement(e, i);
      ponos::point3 b3 =
          positionElement(e, (i + 1) % meshDescriptor.elementSize);
      ponos::point3 c3 =
          positionElement(e, (i + 2) % meshDescriptor.elementSize);
      ponos::point2 a(a3.x, a3.y);
      ponos::point2 b(b3.x, b3.y);
      ponos::point2 c(c3.x, c3.y);
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

void RawMesh::clear() {
  meshDescriptor.elementSize = meshDescriptor.count =
      positionDescriptor.elementSize = positionDescriptor.count =
          normalDescriptor.elementSize = normalDescriptor.count =
              texcoordDescriptor.elementSize = texcoordDescriptor.count =
                  interleavedDescriptor.elementSize =
                      interleavedDescriptor.count = 0;
  indices.clear();
  positions.clear();
  normals.clear();
  texcoords.clear();
  positionsIndices.clear();
  normalsIndices.clear();
  texcoordsIndices.clear();
  interleavedData.clear();
}

void fitToBBox(RawMesh *rm, const bbox2 &bbox) {
  UNUSED_VARIABLE(bbox);
  float ratio = rm->bbox.size(1) / rm->bbox.size(0);
  if (rm->bbox.size(0) > rm->bbox.size(1))
    rm->apply(
        scale(1.f / rm->bbox.size(0), ratio * (1.f / rm->bbox.size(0)), 0) *
        translate(point3() - rm->bbox.lower));
  else
    rm->apply(
        scale((1.f / rm->bbox.size(1)) / ratio, 1.f / rm->bbox.size(1), 0) *
        translate(point3() - rm->bbox.lower));
  rm->computeBBox();
}

std::ostream &operator<<(std::ostream &os, RawMesh &rm) {
  os << "RawMesh:\n";
  os << "vertex description (dim = " << rm.positionDescriptor.elementSize
     << ", count_ = " << rm.positionDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.positionDescriptor.count; i++) {
    os << "v" << i << " = ";
    for (size_t j = 0; j < rm.positionDescriptor.elementSize; j++)
      os << rm.positions[i * rm.positionDescriptor.elementSize + j] << " ";
    os << std::endl;
  }
  os << "normal description (dim = " << rm.normalDescriptor.elementSize
     << ", count_ = " << rm.normalDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.normalDescriptor.count; i++) {
    os << "v" << i << " = ";
    for (size_t j = 0; j < rm.normalDescriptor.elementSize; j++)
      os << rm.normals[i * rm.normalDescriptor.elementSize + j] << " ";
    os << std::endl;
  }
  os << "texcoord description (dim = " << rm.texcoordDescriptor.elementSize
     << ", count_ = " << rm.texcoordDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.texcoordDescriptor.count; i++) {
    os << "t" << i << " = ";
    for (size_t j = 0; j < rm.texcoordDescriptor.elementSize; j++)
      os << rm.texcoords[i * rm.texcoordDescriptor.elementSize + j] << " ";
    os << std::endl;
  }
  os << "mesh description (dim = " << rm.meshDescriptor.elementSize
     << ", count_ = " << rm.meshDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.meshDescriptor.count; i++) {
    os << "m" << i << " = ";
    for (size_t j = 0; j < rm.meshDescriptor.elementSize; j++)
      os << "("
         << rm.indices[i * rm.meshDescriptor.elementSize + j].positionIndex
         << ", "
         << rm.indices[i * rm.meshDescriptor.elementSize + j].normalIndex
         << ", "
         << rm.indices[i * rm.meshDescriptor.elementSize + j].texcoordIndex
         << ")";
    os << std::endl;
  }
  os << "indices (dim = " << rm.meshDescriptor.elementSize
     << ", count = " << rm.meshDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.meshDescriptor.count; i++) {
    for (size_t j = 0; j < rm.meshDescriptor.elementSize; j++)
      os << rm.positionsIndices[i * rm.meshDescriptor.elementSize + j] << " ";
    os << std::endl;
  }
  os << "interleaved description (dim = "
     << rm.interleavedDescriptor.elementSize
     << ", count_ = " << rm.interleavedDescriptor.count << ")\n";
  for (size_t i = 0; i < rm.interleavedDescriptor.count; i++) {
    os << "i" << i << " = ";
    for (size_t j = 0; j < rm.interleavedDescriptor.elementSize; j++)
      os << rm.interleavedData[i * rm.interleavedDescriptor.elementSize + j]
         << " ";
    os << std::endl;
  }
  return os;
}
} // namespace ponos
