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

//
// Created by FilipeCN on 2/19/2018.
//

#include <circe/gl/scene/instance_set.h>

#include <utility>

namespace circe::gl {

InstanceSet::InstanceSet(SceneMesh &m, ShaderProgram s, size_t n)
    : shader_(s), baseMesh_(m) {
  if (n)
    resize(n);
}

InstanceSet::~InstanceSet() {
  for (auto &buffer : buffers_)
    delete buffer;
}

uint InstanceSet::add(BufferDescriptor d) {
  d.elementCount = count_;
  size_t b = buffers_.size();
  size_t bs = d.elementCount * d.elementSize;
  switch (d.dataType) {
  case GL_UNSIGNED_INT: {
    buffers_.emplace_back(new Buffer<uint>(d));
    dataU_.emplace_back(std::vector<uint>());
    buffersIndices_.emplace_back(dataU_.size() - 1);
    if (count_)
      dataU_[dataU_.size() - 1].resize(bs);
    break;
  }
  case GL_UNSIGNED_BYTE: {
    buffers_.emplace_back(new Buffer<uint>(d));
    dataC_.emplace_back(std::vector<uchar>());
    buffersIndices_.emplace_back(dataC_.size() - 1);
    if (count_)
      dataC_[dataC_.size() - 1].resize(bs);
    break;
  }
  default: {
    buffers_.emplace_back(new Buffer<float>(d));
    dataF_.emplace_back(std::vector<float>());
    buffersIndices_.emplace_back(dataF_.size() - 1);
    if (count_)
      dataF_[dataF_.size() - 1].resize(bs);
  }
  }
  //  for (auto &attribute : d.attributes)
  //    shader_.addVertexAttribute(attribute.first.c_str());
  return b;
}

void InstanceSet::resize(uint n) {
  count_ = n;
  for (uint i = 0; i < buffers_.size(); i++) {
    buffers_[i]->bufferDescriptor.elementCount = n;
    switch (buffers_[i]->bufferDescriptor.dataType) {
    case GL_UNSIGNED_INT:
      dataU_[buffersIndices_[i]].resize(
          buffers_[i]->bufferDescriptor.elementCount *
          buffers_[i]->bufferDescriptor.elementSize);
      dynamic_cast<Buffer<uint> *>(buffers_[i])
          ->set(&dataU_[buffersIndices_[i]][0], buffers_[i]->bufferDescriptor);
      break;
    case GL_UNSIGNED_BYTE:
      dataC_[buffersIndices_[i]].resize(
          buffers_[i]->bufferDescriptor.elementCount *
          buffers_[i]->bufferDescriptor.elementSize);
      dynamic_cast<Buffer<uchar> *>(buffers_[i])
          ->set(&dataC_[buffersIndices_[i]][0], buffers_[i]->bufferDescriptor);
      break;
    default:
      dataF_[buffersIndices_[i]].resize(
          buffers_[i]->bufferDescriptor.elementCount *
          buffers_[i]->bufferDescriptor.elementSize);
      dynamic_cast<Buffer<float> *>(buffers_[i])
          ->set(&dataF_[buffersIndices_[i]][0], buffers_[i]->bufferDescriptor);
    }
  }
  dataChanged_.resize(n, false);
}

float *InstanceSet::instanceF(uint b, uint i) {
  if (i >= count_)
    resize(i + 1);
  dataChanged_[b] = true;
  return &dataF_[buffersIndices_[b]]
                [i * buffers_[b]->bufferDescriptor.elementSize];
}

uint *InstanceSet::instanceU(uint b, uint i) {
  if (i >= count_)
    resize(i + 1);
  dataChanged_[b] = true;
  return &dataU_[buffersIndices_[b]]
                [i * buffers_[b]->bufferDescriptor.elementSize];
}

void InstanceSet::bind(uint b) {
  if (dataChanged_[b]) {
    switch (buffers_[b]->bufferDescriptor.dataType) {
    case GL_UNSIGNED_INT:
      dynamic_cast<Buffer<uint> *>(buffers_[b])
          ->set(&dataU_[buffersIndices_[b]][0]);
      break;
    case GL_UNSIGNED_BYTE:
      dynamic_cast<Buffer<uchar> *>(buffers_[b])
          ->set(&dataC_[buffersIndices_[b]][0]);
      break;
    default:
      dynamic_cast<Buffer<float> *>(buffers_[b])
          ->set(&dataF_[buffersIndices_[b]][0]);
    }
    dataChanged_[b] = false;
  }
  buffers_[b]->bind();
}

void InstanceSet::draw(const CameraInterface *camera,
                       ponos::Transform transform) {
  UNUSED_VARIABLE(transform);
  // bind buffers and locate attributes
  shader_.begin();
  baseMesh_.bind();
  baseMesh_.vertexBuffer()->locateAttributes(shader_);
  for (size_t i = 0; i < buffers_.size(); i++) {
    bind(i);
    buffers_[i]->locateAttributes(shader_, 1);
  }
  shader_.setUniform("model_view_matrix",
                     ponos::transpose(camera->getViewTransform().matrix()));
  shader_.setUniform(
      "projection_matrix",
      ponos::transpose(camera->getProjectionTransform().matrix()));
  CHECK_GL_ERRORS;
  //  shader_.setUniform("mvp",
  //  ponos::transpose((camera->getProjectionTransform() *
  //      camera->getViewTransform() * camera->getModelTransform()).matrix()));
  glDrawElementsInstanced(
      baseMesh_.indexBuffer()->bufferDescriptor.elementType,
      baseMesh_.indexBuffer()->bufferDescriptor.elementSize *
          baseMesh_.indexBuffer()->bufferDescriptor.elementCount,
      baseMesh_.indexBuffer()->bufferDescriptor.dataType, 0, count_);
  CHECK_GL_ERRORS;
  shader_.end();
  baseMesh_.unbind();
}

} // namespace circe
