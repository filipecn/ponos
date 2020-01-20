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

RawMesh *create_icosphere_mesh(const point3 &center, float radius,
                               size_t divisions, bool generateNormals,
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
  {
    // normalize positions
    size_t n = mesh->positions.size() / 3;
    for (size_t i = 0; i < n; i++) {
      vec3 v(&mesh->positions[i * 3]);
      v = normalize(v);
      mesh->positions[i * 3 + 0] = v.x;
      mesh->positions[i * 3 + 1] = v.y;
      mesh->positions[i * 3 + 2] = v.z;
    }
  }
  std::map<std::pair<int, int>, size_t> indicesCache;
  std::function<size_t(int, int)> midPoint = [&](int a, int b) -> size_t {
    std::pair<int, int> key(std::min(a, b), std::max(a, b));
    size_t n = mesh->positions.size() / 3;
    if (indicesCache.find(key) != indicesCache.end())
      return indicesCache[key];
    point3 pa(&mesh->positions[3 * a]);
    point3 pb(&mesh->positions[3 * b]);
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
    vec3 v(&mesh->positions[i * 3]);
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
    std::function<point2(float, float, float)> parametric =
        [](float x, float y, float z) -> point2 {
          return point2(std::atan2(y, x), std::acos(z));
        };
    for (size_t i = 0; i < vertexCount; i++) {
      auto uv =
          parametric(mesh->positions[i * 3 + 0], mesh->positions[i * 3 + 1],
                     mesh->positions[i * 3 + 2]);
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
  mesh->apply(scale(radius, radius, radius) * translate(vec3(center)));
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMesh *create_quad_mesh(const point3 &p1, const point3 &p2, const point3 &p3,
                          const point3 &p4, bool generateNormals,
                          bool generateUVs) {
  auto *mesh = new RawMesh();
  mesh->addPosition({p1.x, p1.y, p1.z});
  mesh->addPosition({p2.x, p2.y, p2.z});
  mesh->addPosition({p3.x, p3.y, p3.z});
  mesh->addPosition({p4.x, p4.y, p4.z});
  mesh->addFace({0, 1, 2});
  mesh->addFace({0, 2, 3});
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 3;
  if (generateNormals) {
    mesh->normals = mesh->positions;
    mesh->normalDescriptor.count = vertexCount;
    mesh->normalDescriptor.elementSize = 3;
    /// TODO
    std::cerr << LOG_LOCATION << "Normals are not being generated yet\n";
  }
  if (generateUVs) {
    mesh->addUV({0.f, 0.f});
    mesh->addUV({1.f, 0.f});
    mesh->addUV({1.f, 1.f});
    mesh->addUV({0.f, 1.f});
    mesh->texcoordDescriptor.count = vertexCount;
    mesh->texcoordDescriptor.elementSize = 2;
  }
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::TRIANGLES;
  mesh->meshDescriptor.count = mesh->indices.size() / 3;
  mesh->meshDescriptor.elementSize = 3;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 3;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return std::move(mesh);
}

RawMesh *create_quad_wireframe_mesh(const point3 &p1, const point3 &p2,
                                    const point3 &p3, const point3 &p4,
                                    bool triangleFaces) {
  auto *mesh = new RawMesh();
  mesh->addPosition({p1.x, p1.y, p1.z});
  mesh->addPosition({p2.x, p2.y, p2.z});
  mesh->addPosition({p3.x, p3.y, p3.z});
  mesh->addPosition({p4.x, p4.y, p4.z});
  if (triangleFaces) {
    mesh->addFace({0, 1, 2});
    mesh->addFace({0, 2, 3});
  } else
    mesh->addFace({0, 1, 2, 3});
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 3;
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::LINE_LOOP;
  mesh->meshDescriptor.count = (triangleFaces) ? 2 : 1;
  mesh->meshDescriptor.elementSize = (triangleFaces) ? 3 : 4;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 3;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMeshSPtr icosphere(const point2 &center, float radius, size_t divisions,
                      bool generateNormals) {
  RawMeshSPtr mesh = std::make_shared<RawMesh>();
  // icosahedron
  float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
  mesh->addPosition({-1, t});
  mesh->addPosition({1, t});
  mesh->addPosition({-1, -t});
  mesh->addPosition({1, -t});
  mesh->addFace({0, 2});
  mesh->addFace({2, 1});
  mesh->addFace({1, 3});
  mesh->addFace({3, 0});
  {
    // normalize positions
    size_t n = mesh->positions.size() / 2;
    for (size_t i = 0; i < n; i++) {
      vec2 v(&mesh->positions[i * 2]);
      v = normalize(v);
      mesh->positions[i * 2 + 0] = v.x;
      mesh->positions[i * 2 + 1] = v.y;
    }
  }
  std::map<std::pair<int, int>, size_t> indicesCache;
  std::function<size_t(int, int)> midPoint = [&](int a, int b) -> size_t {
    std::pair<int, int> key(std::min(a, b), std::max(a, b));
    size_t n = mesh->positions.size() / 2;
    if (indicesCache.find(key) != indicesCache.end())
      return indicesCache[key];
    point2 pa(&mesh->positions[2 * a]);
    point2 pb(&mesh->positions[2 * b]);
    auto pm = pa + (pb - pa) * 0.5f;
    mesh->addPosition({pm.x, pm.y});
    return n;
  };
  for (size_t i = 0; i < divisions; i++) {
    size_t n = mesh->indices.size() / 2;
    for (size_t j = 0; j < n; j++) {
      int v0 = mesh->indices[j * 2 + 0].positionIndex;
      int v1 = mesh->indices[j * 2 + 1].positionIndex;
      int a = midPoint(v0, v1);
      mesh->addFace({v0, a});
      mesh->addFace({a, v0});
    }
  }
  // normalize positions
  size_t n = mesh->positions.size() / 2;
  for (size_t i = 0; i < n; i++) {
    vec2 v(&mesh->positions[i * 2]);
    v = normalize(v);
    mesh->positions[i * 2 + 0] = v.x;
    mesh->positions[i * 2 + 1] = v.y;
  }
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 2;
  if (generateNormals) {
    mesh->normals = mesh->positions;
    mesh->normalDescriptor.count = vertexCount;
    mesh->normalDescriptor.elementSize = 2;
  }
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::LINES;
  mesh->meshDescriptor.count = mesh->indices.size() / 2;
  mesh->meshDescriptor.elementSize = 2;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 2;
  mesh->apply(scale(radius, radius, 0) *
      translate(vec3(center.x, center.y, 0)));
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMeshSPtr icosphere(const point3 &center, float radius, size_t divisions,
                      bool generateNormals, bool generateUVs) {
  RawMeshSPtr mesh = std::make_shared<RawMesh>();
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
  {
    // normalize positions
    size_t n = mesh->positions.size() / 3;
    for (size_t i = 0; i < n; i++) {
      vec3 v(&mesh->positions[i * 3]);
      v = normalize(v);
      mesh->positions[i * 3 + 0] = v.x;
      mesh->positions[i * 3 + 1] = v.y;
      mesh->positions[i * 3 + 2] = v.z;
    }
  }
  std::map<std::pair<int, int>, size_t> indicesCache;
  std::function<size_t(int, int)> midPoint = [&](int a, int b) -> size_t {
    std::pair<int, int> key(std::min(a, b), std::max(a, b));
    size_t n = mesh->positions.size() / 3;
    if (indicesCache.find(key) != indicesCache.end())
      return indicesCache[key];
    point3 pa(&mesh->positions[3 * a]);
    point3 pb(&mesh->positions[3 * b]);
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
    vec3 v(&mesh->positions[i * 3]);
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
    std::function<point2(float, float, float)> parametric =
        [](float x, float y, float z) -> point2 {
          return point2(std::atan2(y, x), std::acos(z));
        };
    for (size_t i = 0; i < vertexCount; i++) {
      auto uv =
          parametric(mesh->positions[i * 3 + 0], mesh->positions[i * 3 + 1],
                     mesh->positions[i * 3 + 2]);
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
  mesh->apply(scale(radius, radius, radius) * translate(vec3(center)));
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMeshSPtr RawMeshes::segment(const point2 &a, const point2 &b) {
  RawMeshSPtr mesh = std::make_shared<RawMesh>();
  mesh->addPosition({a.x, a.y});
  mesh->addPosition({b.x, b.y});
  mesh->addFace({0, 1});
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 2;
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::LINES;
  mesh->meshDescriptor.count = mesh->indices.size() / 2;
  mesh->meshDescriptor.elementSize = 2;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 2;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMeshSPtr RawMeshes::segment(const point3 &a, const point3 &b) {
  RawMeshSPtr mesh = std::make_shared<RawMesh>();
  mesh->addPosition({a.x, a.y, a.z});
  mesh->addPosition({b.x, b.y, b.z});
  mesh->addFace({0, 1});
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 3;
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::LINES;
  mesh->meshDescriptor.count = mesh->indices.size() / 2;
  mesh->meshDescriptor.elementSize = 2;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 3;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMesh *RawMeshes::cube(const ponos::Transform &transform,
                         bool generateNormals, bool generateUVs) {
  auto *mesh = new RawMesh();
  //  y          2      3
  //  |_x     6      7
  // z          0       1
  //          4      5
  mesh->addPosition({0.f, 0.f, 0.f});
  mesh->addPosition({1.f, 0.f, 0.f});
  mesh->addPosition({0.f, 1.f, 0.f});
  mesh->addPosition({1.f, 1.f, 0.f});
  mesh->addPosition({0.f, 0.f, 1.f});
  mesh->addPosition({1.f, 0.f, 1.f});
  mesh->addPosition({0.f, 1.f, 1.f});
  mesh->addPosition({1.f, 1.f, 1.f});
  mesh->addFace({0, 2, 1}); // back
  mesh->addFace({1, 2, 3}); // back
  mesh->addFace({4, 5, 6}); // front
  mesh->addFace({6, 5, 7}); // front
  mesh->addFace({0, 4, 6}); // left
  mesh->addFace({0, 6, 2}); // left
  mesh->addFace({5, 1, 7}); // right
  mesh->addFace({1, 3, 7}); // right
  mesh->addFace({0, 1, 5}); // bottom
  mesh->addFace({0, 5, 4}); // bottom
  mesh->addFace({2, 7, 3}); // top
  mesh->addFace({2, 6, 7}); // top
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 3;
  if (generateNormals) {
    mesh->normals = mesh->positions;
    mesh->normalDescriptor.count = vertexCount;
    mesh->normalDescriptor.elementSize = 3;
    // TODO
    std::cerr << LOG_LOCATION << "Normals are not being generated yet\n";
  }
  if (generateUVs) {
    mesh->addUV({0, 0, 0}); // 0
    mesh->addUV({1, 0, 0}); // 1
    mesh->addUV({0, 1, 0}); // 2
    mesh->addUV({1, 1, 0}); // 3
    mesh->addUV({0, 0, 1}); // 4
    mesh->addUV({1, 0, 1}); // 5
    mesh->addUV({0, 1, 1}); // 6
    mesh->addUV({1, 1, 1}); // 7
    mesh->texcoordDescriptor.count = vertexCount;
    mesh->texcoordDescriptor.elementSize = 3;
  }
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::TRIANGLES;
  mesh->meshDescriptor.count = mesh->indices.size() / 3;
  mesh->meshDescriptor.elementSize = 3;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 3;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMesh *RawMeshes::quad(const ponos::Transform &transform,
                         bool generate_uvs) {
  auto *mesh = new RawMesh();
  mesh->addPosition({0.f, 0.f});
  mesh->addPosition({1.f, 0.f});
  mesh->addPosition({1.f, 1.f});
  mesh->addPosition({0.f, 1.f});
  mesh->addFace({0, 1, 2});
  mesh->addFace({0, 2, 3});
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  size_t vertexCount = mesh->positions.size() / 2;
  if (generate_uvs) {
    mesh->addUV({0, 0}); // 0
    mesh->addUV({1, 0}); // 1
    mesh->addUV({1, 1}); // 2
    mesh->addUV({0, 1}); // 3
    mesh->texcoordDescriptor.count = vertexCount;
    mesh->texcoordDescriptor.elementSize = 2;
  }
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::TRIANGLES;
  mesh->meshDescriptor.count = mesh->indices.size() / 3;
  mesh->meshDescriptor.elementSize = 3;
  mesh->positionDescriptor.count = vertexCount;
  mesh->positionDescriptor.elementSize = 2;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

RawMeshSPtr RawMeshes::cubeWireframe(const Transform &transform,
                                     bool triangleFaces) {
  RawMeshSPtr mesh = std::make_shared<RawMesh>();
  //  y          2      3
  //  |_x     6      7
  // z          0       1
  //          4      5
  mesh->addPosition({0.f, 0.f, 0.f});
  mesh->addPosition({1.f, 0.f, 0.f});
  mesh->addPosition({0.f, 1.f, 0.f});
  mesh->addPosition({1.f, 1.f, 0.f});
  mesh->addPosition({0.f, 0.f, 1.f});
  mesh->addPosition({1.f, 0.f, 1.f});
  mesh->addPosition({0.f, 1.f, 1.f});
  mesh->addPosition({1.f, 1.f, 1.f});
  if (triangleFaces) {
    mesh->addFace({0, 2, 2, 1}); // back
    mesh->addFace({1, 2, 2, 3}); // back
    mesh->addFace({4, 5, 5, 6}); // front
    mesh->addFace({6, 5, 5, 7}); // front
    mesh->addFace({0, 4, 4, 6}); // left
    mesh->addFace({0, 6, 6, 2}); // left
    mesh->addFace({5, 1, 1, 7}); // right
    mesh->addFace({1, 3, 3, 7}); // right
    mesh->addFace({0, 1, 1, 5}); // bottom
    mesh->addFace({0, 5, 5, 4}); // bottom
  } else {
    mesh->addFace({0, 1});
    mesh->addFace({0, 2});
    mesh->addFace({0, 4});
    mesh->addFace({6, 2});
    mesh->addFace({6, 7});
    mesh->addFace({6, 4});
    mesh->addFace({3, 2});
    mesh->addFace({3, 7});
    mesh->addFace({3, 1});
    mesh->addFace({5, 4});
    mesh->addFace({5, 7});
    mesh->addFace({5, 1});
  }
  // fix normal and uvs indices
  for (auto &i : mesh->indices)
    i.normalIndex = i.texcoordIndex = i.positionIndex;
  // describe mesh
  mesh->primitiveType = GeometricPrimitiveType::LINES;
  mesh->meshDescriptor.count = mesh->indices.size() / 2;
  mesh->meshDescriptor.elementSize = 2;
  mesh->positionDescriptor.count = mesh->positions.size() / 3;
  mesh->positionDescriptor.elementSize = 3;
  mesh->computeBBox();
  mesh->splitIndexData();
  mesh->buildInterleavedData();
  return mesh;
}

} // namespace ponos
