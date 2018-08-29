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

#include <aergia/scene/quad.h>

namespace aergia {

Quad::Quad() {
  this->rawMesh = new ponos::RawMesh();
  this->rawMesh->meshDescriptor.elementSize = 3;
  this->rawMesh->meshDescriptor.count = 2;
  this->rawMesh->positionDescriptor.elementSize = 2;
  this->rawMesh->positionDescriptor.count = 4;
  this->rawMesh->texcoordDescriptor.elementSize = 2;
  this->rawMesh->texcoordDescriptor.count = 4;
  this->rawMesh->positions = std::vector<float>({-1, -1, 1, -1, 1, 1, -1, 1});
  this->rawMesh->texcoords = std::vector<float>({0, 1, 1, 1, 1, 0, 0, 0});
  this->rawMesh->indices.resize(6);
  this->rawMesh->indices[0].positionIndex = this->rawMesh->indices[0].texcoordIndex = 0;
  this->rawMesh->indices[1].positionIndex = this->rawMesh->indices[1].texcoordIndex = 1;
  this->rawMesh->indices[2].positionIndex = this->rawMesh->indices[2].texcoordIndex = 2;
  this->rawMesh->indices[3].positionIndex = this->rawMesh->indices[3].texcoordIndex = 0;
  this->rawMesh->indices[4].positionIndex = this->rawMesh->indices[4].texcoordIndex = 2;
  this->rawMesh->indices[5].positionIndex = this->rawMesh->indices[5].texcoordIndex = 3;
  this->rawMesh->splitIndexData();
  this->rawMesh->buildInterleavedData();
  const char *fs = "#version 440 core\n"
      "in vec2 texCoord;"
      "out vec4 outColor;"
      "layout (location = 1) uniform sampler2D tex;"
      "void main(){"
      " outColor = vec4(1,0,0,1);}";
  const char *vs = "#version 440 core\n"
      "layout (location = 0) in vec2 position;"
      "layout (location = 1) in vec2 texcoord;"
      "layout (location = 0) uniform mat4 mvp;"
      "out vec2 texCoord;"
      "void main() {"
      " texCoord = texcoord;"
      "gl_Position = mvp * vec4(position, 0, 1);}";
  shader.reset(new ShaderProgram(vs, nullptr, fs));
  shader->addVertexAttribute("position", 0);
  shader->addVertexAttribute("texcoord", 1);
  shader->addUniform("mvp", 0);
  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);
  BufferDescriptor vd, id;
  create_buffer_description_from_mesh(*this->rawMesh, vd, id);
  vb.reset(new VertexBuffer(&this->rawMesh->interleavedData[0], vd));
  ib.reset(new IndexBuffer(&this->rawMesh->positionsIndices[0], id));
  vb->locateAttributes(*shader.get());
//  shader->registerVertexAttributes(vb.get());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  CHECK_GL_ERRORS;
//  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
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

void Quad::draw(const CameraInterface *camera, ponos::Transform transform) {
  glBindVertexArray(VAO);
  shader->begin();
  shader->setUniform("mvp", ponos::transpose((camera->getProjectionTransform() *
      camera->getViewTransform() * camera->getModelTransform()).matrix()));
  glDrawElements(GL_TRIANGLES, ib->bufferDescriptor.elementSize *
      ib->bufferDescriptor.elementCount, GL_UNSIGNED_INT, 0);
  CHECK_GL_ERRORS;
  shader->end();
  glBindVertexArray(0);
}

} // aergia nanespace
