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

aergia::SceneMesh::SceneMesh(ponos::RawMesh &rm) : mesh_(rm) {
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
  aergia::BufferDescriptor ver, ind;
  aergia::create_buffer_description_from_mesh(mesh_, ver, ind);
  // TODO
  ver.elementSize = ((rm.positionDescriptor.count) ? rm.positionDescriptor.elementSize : 0) +
      ((rm.normalDescriptor.count) ? rm.normalDescriptor.elementSize : 0) +
      ((rm.texcoordDescriptor.count) ? rm.texcoordDescriptor.elementSize : 0);
  ver.elementCount = vertexData_.size() / ver.elementSize;
  ver.addAttribute(std::string("position"),
                   mesh_.positionDescriptor.elementSize, 0,
                   GL_FLOAT);
  size_t offset = mesh_.positionDescriptor.elementSize;
  if (mesh_.normalDescriptor.count) {
    ver.addAttribute(std::string("normal"),
                     mesh_.normalDescriptor.elementSize,
                     offset * sizeof(float), GL_FLOAT);
    offset += mesh_.normalDescriptor.elementSize;
  }
  if (mesh_.texcoordDescriptor.count) {
    ver.addAttribute(std::string("texcoord"),
                     mesh_.texcoordDescriptor.elementSize,
                     offset * sizeof(float), GL_FLOAT);
    offset += mesh_.texcoordDescriptor.elementSize;
  }
  vertexBuffer_.set(&vertexData_[0], ver);
  indexBuffer_.set(&indexData_[0], ind);
}

void aergia::SceneMesh::bind() {
  vertexBuffer_.bind();
  indexBuffer_.bind();
}

const aergia::VertexBuffer *aergia::SceneMesh::vertexBuffer() const {
  return &vertexBuffer_;
}

const aergia::IndexBuffer *aergia::SceneMesh::indexBuffer() const {
  return &indexBuffer_;
}
