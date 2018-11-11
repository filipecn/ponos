// Created by FilipeCN on 2/21/2018.
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

#include "scene_mesh.h"

circe::SceneMesh::SceneMesh(ponos::RawMesh &rm) : mesh_(rm) {
  { // convert mesh to buffers
    // position index | norm index | texcoord index -> index
    typedef std::pair<std::pair<int, int>, int> MapKey;
    std::map<MapKey, size_t> m;
    size_t newIndex = 0;
    for (auto &i : rm.indices) {
      auto key = MapKey(std::pair<int, int>(i.positionIndex, i.normalIndex),
                        i.texcoordIndex);
      auto it = m.find(key);
      auto index = newIndex;
      if (it == m.end()) {
        // add data
        if (rm.positionDescriptor.count)
          for (size_t v = 0; v < rm.positionDescriptor.elementSize; v++)
            vertexData_.emplace_back(
                rm.positions[i.positionIndex *
                                 rm.positionDescriptor.elementSize +
                             v]);
        if (rm.normalDescriptor.count)
          for (size_t n = 0; n < rm.normalDescriptor.elementSize; n++)
            vertexData_.emplace_back(
                rm.normals[i.normalIndex * rm.normalDescriptor.elementSize +
                           n]);
        if (rm.texcoordDescriptor.count)
          for (size_t t = 0; t < rm.texcoordDescriptor.elementSize; t++)
            vertexData_.emplace_back(
                rm.texcoords[i.texcoordIndex *
                                 rm.texcoordDescriptor.elementSize +
                             t]);
        m[key] = newIndex++;
      } else
        index = it->second;
      indexData_.emplace_back(index);
    }
  }
  circe::BufferDescriptor ver, ind;
  circe::create_buffer_description_from_mesh(mesh_, ver, ind);
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);
  vertexBuffer_.set(&vertexData_[0], ver);
  indexBuffer_.set(&indexData_[0], ind);
  //  vertexBuffer_.set(&mesh_.interleavedData[0], ver);
  //  indexBuffer_.set(&mesh_.positionsIndices[0], ind);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  CHECK_GL_ERRORS;
}

void circe::SceneMesh::bind() {
  glBindVertexArray(VAO);
  vertexBuffer_.bind();
  indexBuffer_.bind();
}

const circe::VertexBuffer *circe::SceneMesh::vertexBuffer() const {
  return &vertexBuffer_;
}

const circe::IndexBuffer *circe::SceneMesh::indexBuffer() const {
  return &indexBuffer_;
}

void circe::SceneMesh::unbind() {
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}
