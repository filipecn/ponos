// Created by filipecn on 3/14/18.
/*
 * Copyright (c) 2018 FilipeCN
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

#include "geometric_shapes.h"
#include <map>

namespace ponos {

RawMesh *create_icosphere_mesh(const Point3 &center,
                               float radius,
                               size_t divisions,
                               bool generateNormals,
                               bool generateUVs) {
  auto *mesh = new RawMesh();
  // icosahedron
  float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
  mesh->addPosition({-1, t, 0});
  mesh->addPosition({1, t, 0});
  mesh->addPosition({-1, -t, 0});
  mesh->addPosition({1, -t, 0});
  mesh->addPosition({0, -1, t});
  mesh->addPosition({0, 1, t});
  mesh->addPosition({0, -1, -t});
  mesh->addPosition({0, 1, -t});
  mesh->addPosition({t, 0, -1});
  mesh->addPosition({t, 0, 1});
  mesh->addPosition({-t, 0, -1});
  mesh->addPosition({-t, 0, 1});
  mesh->addFace({0, 11, 5});
  mesh->addFace({0, 5, 1});
  mesh->addFace({0, 1, 7});
  mesh->addFace({0, 7, 10});
  mesh->addFace({0, 10, 11});
  mesh->addFace({1, 5, 9});
  mesh->addFace({5, 11, 4});
  mesh->addFace({11, 10, 2});
  mesh->addFace({10, 7, 6});
  mesh->addFace({7, 1, 8});
  mesh->addFace({3, 9, 4});
  mesh->addFace({3, 4, 2});
  mesh->addFace({3, 2, 6});
  mesh->addFace({3, 6, 8});
  mesh->addFace({3, 8, 9});
  mesh->addFace({4, 9, 5});
  mesh->addFace({2, 4, 11});
  mesh->addFace({6, 2, 10});
  mesh->addFace({8, 6, 7});
  mesh->addFace({9, 8, 1});
  std::map<std::pair<int, int>, size_t> indicesCache;
  std::function<size_t(int, int)> midPoint = [&](int a, int b) -> size_t {
    std::pair<int, int> key(std::min(a, b), std::max(a, b));
    size_t n = mesh->positions.size() / 3;
    if (indicesCache.find(key) != indicesCache.end())
      return indicesCache[key];
    Point3 pa(&mesh->positions[3 * a]);
    Point3 pb(&mesh->positions[3 * b]);
    auto pm = pa + (pb - pa) * 0.5f;
    mesh->addPosition({pm.x, pm.y, pm.z});
    return n;
  };
  for (size_t i = 0; i < divisions; i++) {
    size_t n = mesh->indices.size() / 3;
    for (size_t j = 0; j < n; j++) {
      int v0 = mesh->indices[j * 3 + 0].positionIndex;
      int v1 = mesh->indices[j * 3 + 1].positionIndex;
      int v2 = mesh->indices[j * 3 + 2].positionIndex;
      int a = midPoint(v0, v1);
      int b = midPoint(v1, v2);
      int c = midPoint(v2, v0);
      mesh->addFace({v0, a, c});
      mesh->addFace({v1, b, a});
      mesh->addFace({v2, c, b});
      mesh->addFace({a, c, b});
    }
  }
  // normalize positions
  size_t n = mesh->positions.size() / 3;
  for (size_t i = 0; i < n; i++) {
    Vector3 v(&mesh->positions[i * 3]);
    v = normalize(v);
    mesh->positions[i * 3 + 0] = v.x;
    mesh->positions[i * 3 + 1] = v.y;
    mesh->positions[i * 3 + 2] = v.z;
  }
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 3;
  if (generateNormals) {
    mesh->normals = mesh->positions;
    mesh->normalDescriptor.count = vertexCount;
    mesh->normalDescriptor.elementSize = 3;
  }
  if (generateUVs) {
    std::function<Point2(float, float, float)> parametric = [](float x, float y, float z) -> Point2 {
      return Point2(std::atan2(y, x), std::acos(z));
    };
    for (size_t i = 0; i < vertexCount; i++) {
      auto uv = parametric(mesh->positions[i * 3 + 0], mesh->positions[i * 3 + 1], mesh->positions[i * 3 + 2]);
      mesh->addUV({uv.x, uv.y});
    }
    mesh->texcoordDescriptor.count = vertexCount;
    mesh->texcoordDescriptor.elementSize = 2;
  }
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::TRIANGLES;
  mesh->meshDescriptor.count = mesh->indices.size() / 3;
  mesh->meshDescriptor.elementSize = 3;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 3;
  mesh->apply(scale(radius, radius, radius) *
      translate(Vector3(center)));
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

} // ponos namespace