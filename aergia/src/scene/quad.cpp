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

#include "scene/quad.h"

namespace aergia {

Quad::Quad() {
  this->rawMesh = new ponos::RawMesh();
  this->rawMesh->meshDescriptor.elementSize = 4;
  this->rawMesh->meshDescriptor.count = 1;
  this->rawMesh->vertexDescriptor.elementSize = 2;
  this->rawMesh->vertexDescriptor.count = 4;
  this->rawMesh->texcoordDescriptor.elementSize = 2;
  this->rawMesh->texcoordDescriptor.count = 4;
  this->rawMesh->vertices = std::vector<float>({-1, -1, 1, -1, 1, 1, -1, 1});
  this->rawMesh->texcoords = std::vector<float>({0, 1, 1, 1, 1, 0, 0, 0});
  this->rawMesh->indices.resize(4);
  for (int i = 0; i < 4; i++)
    this->rawMesh->indices[i].vertexIndex = rawMesh->indices[i].texcoordIndex =
        i;
  this->rawMesh->splitIndexData();
  this->rawMesh->buildInterleavedData();
  glGenVertexArrays(1, &VAO);
  setupVertexBuffer(GL_TRIANGLES, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
  setupIndexBuffer();
}

void Quad::set(const ponos::Point2 &pm, const ponos::Point2 &pM) {
  this->rawMesh->interleavedData[0] = pm.x;
  this->rawMesh->interleavedData[1] = pm.y;
  this->rawMesh->interleavedData[4] = pM.x;
  this->rawMesh->interleavedData[5] = pm.y;
  this->rawMesh->interleavedData[8] = pM.x;
  this->rawMesh->interleavedData[9] = pM.y;
  this->rawMesh->interleavedData[12] = pm.x;
  this->rawMesh->interleavedData[13] = pM.y;
  glBindVertexArray(VAO);
  this->vb->set(&this->rawMesh->interleavedData[0]);
  glBindVertexArray(0);
}

void Quad::draw() const {
  glBindVertexArray(VAO);
  vb->bind();
  ib->bind();
  shader->begin(vb.get());
  glDrawElements(GL_QUADS, ib->bufferDescriptor.elementCount, GL_UNSIGNED_INT,
                 0);
  shader->end();
  glBindVertexArray(0);
}

} // aergia nanespace
