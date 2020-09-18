// Created by FilipeCN on 2/22/2018.
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

#include <circe/gl/io/buffer.h>

#include <circe/gl/graphics/shader.h>

#include <utility>

namespace circe::gl {

BufferInterface::BufferInterface(GLuint id) : bufferId(id) {}

BufferInterface::BufferInterface(BufferDescriptor b, GLuint id)
    : bufferDescriptor(std::move(b)), bufferId(id) {}

BufferInterface::~BufferInterface() { glDeleteBuffers(1, &bufferId); }

void BufferInterface::bind() const {
  glBindBuffer(bufferDescriptor.type, bufferId);
}

void BufferInterface::registerAttribute(const std::string &name, GLint location) const {
  if (bufferDescriptor.attribute_map.find(name) ==
      bufferDescriptor.attribute_map.end())
    return;
  const BufferDescriptor::Attribute &va =
      bufferDescriptor.attribute_map.find(name)->second;
  glVertexAttribPointer(static_cast<GLuint>(location), va.size, va.type,
                        GL_FALSE,
                        bufferDescriptor.element_size * sizeof(float),
                        reinterpret_cast<const void *>(va.offset));
}

void BufferInterface::locateAttributes(const ShaderProgram &s, uint d) const {
  for (auto &attribute : bufferDescriptor.attribute_map) {
    GLint loc = s.locateAttribute(attribute.first);
    if (loc < 0)
      continue;
    // in case the attribute is a matrix, we need to point to n vectors
    size_t componentSize = 1;
    if (attribute.second.size % 3 == 0)
      componentSize = 3;
    else if (attribute.second.size % 4 == 0)
      componentSize = 4;
    else if (attribute.second.size % 2 == 0)
      componentSize = 2;
    size_t n = attribute.second.size / componentSize;
    FATAL_ASSERT(n != 0);
    for (size_t i = 0; i < n; i++) {
      glEnableVertexAttribArray(static_cast<GLuint>(loc + i));
      glVertexAttribPointer(static_cast<GLuint>(loc + i), componentSize,
                            attribute.second.type, GL_FALSE,
                            bufferDescriptor.element_size * sizeof(float),
                            (void *) (attribute.second.offset +
                                i * componentSize * sizeof(float)));
      CHECK_GL_ERRORS;
      glVertexAttribDivisor(static_cast<GLuint>(loc + i), d);
    }
  }
}

void BufferInterface::locateAttributes(const Program &s, uint d) const {
  for (auto &attribute : bufferDescriptor.attribute_map) {
    GLint loc = s.locateAttribute(attribute.first);
    if (loc < 0)
      continue;
    // in case the attribute is a matrix, we need to point to n vectors
    size_t componentSize = 1;
    if (attribute.second.size % 3 == 0)
      componentSize = 3;
    else if (attribute.second.size % 4 == 0)
      componentSize = 4;
    else if (attribute.second.size % 2 == 0)
      componentSize = 2;
    size_t n = attribute.second.size / componentSize;
    FATAL_ASSERT(n != 0);
    for (size_t i = 0; i < n; i++) {
      glEnableVertexAttribArray(static_cast<GLuint>(loc + i));
      glVertexAttribPointer(static_cast<GLuint>(loc + i), componentSize,
                            attribute.second.type, GL_FALSE,
                            bufferDescriptor.element_size * sizeof(float),
                            (void *) (attribute.second.offset +
                                i * componentSize * sizeof(float)));
      CHECK_GL_ERRORS;
      glVertexAttribDivisor(static_cast<GLuint>(loc + i), d);
    }
  }
}

GLuint BufferInterface::id() const { return bufferId; }

DeviceMemory::View::View(DeviceMemory &buffer, const BufferDescriptor &descriptor, u64 offset)
    : buffer_(buffer), descriptor_(descriptor), offset_(offset) {
  length_ = std::min(offset_ + descriptor.size(), buffer.size()) - offset_;
}

void *DeviceMemory::View::mapped(GLbitfield access) {
  mapped_ = buffer_.mapped(offset_, length_, access);
  return mapped_;
}

void DeviceMemory::View::unmap() {
  buffer_.unmap();
  mapped_ = nullptr;
}

DeviceMemory::DeviceMemory() {
  glGenBuffers(1, &buffer_object_id_);
}

DeviceMemory::DeviceMemory(GLuint usage, GLuint target, u64 size, void *data) : DeviceMemory() {
  setUsage(usage);
  setTarget(target);
  resize(size);
}

DeviceMemory::~DeviceMemory() {
  destroy();
}

void DeviceMemory::setTarget(GLuint _target) {
  target_ = _target;
}

void DeviceMemory::setUsage(GLuint _usage) {
  usage_ = _usage;
}

void DeviceMemory::resize(u64 size_in_bytes) {
  size_ = size_in_bytes;
}

void DeviceMemory::allocate_(void *data) {
  destroy();
  glCreateBuffers(1, &buffer_object_id_);
  glBindBuffer(target_, buffer_object_id_);
  CHECK_GL(glBufferData(target_, size_, data, usage_));
}

void DeviceMemory::allocate() {
  allocate_(nullptr);
}

void DeviceMemory::allocate(u64 data_size, void *data) {
  resize(data_size);
  allocate_(data);
}

void DeviceMemory::copy(void *data, u64 data_size, u64 offset) {
  bind();
  CHECK_GL(glBufferSubData(target_, offset, data_size, data));
}

void DeviceMemory::bind() {
  if (!allocated())
    allocate();
  glBindBuffer(target_, buffer_object_id_);
}

void *DeviceMemory::mapped(GLenum access) {
  bind();
  void *mapped = glMapBuffer(target_, access);
  CHECK_GL_ERRORS;
  return mapped;
}

void *DeviceMemory::mapped(u64 offset, u64 length, GLbitfield access) {
  bind();
  void *mapped = glMapBufferRange(target_, offset, length, access);
  CHECK_GL_ERRORS;
  return mapped;
}

void DeviceMemory::unmap() const {
  CHECK_GL(glUnmapBuffer(target_));
}

void DeviceMemory::destroy() {
  if (buffer_object_id_) CHECK_GL(glDeleteBuffers(1, &buffer_object_id_));
  buffer_object_id_ = 0;
}

Buffer::Buffer() = default;

Buffer::Buffer(const BufferDescriptor &descriptor) {
  setDescriptor(descriptor);
}

Buffer::~Buffer() = default;

void Buffer::setDescriptor(const BufferDescriptor &descriptor) {
  mem_.setTarget(descriptor.type);
  mem_.setUsage(descriptor.use);
  mem_.resize(descriptor.size());
  view_ = std::make_unique<DeviceMemory::View>(mem_, descriptor);
}

void Buffer::bind() {
  mem_.bind();
}

void *Buffer::mapped(GLbitfield access) {
  return mem_.mapped(access);
}

void Buffer::unmap() const {
  mem_.unmap();
}

} // namespace circe
