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

namespace circe::gl {

BufferInterface::BufferInterface(GLuint id) : bufferId(id) {}

BufferInterface::BufferInterface(BufferDescriptor b, GLuint id)
    : bufferDescriptor(std::move(b)), bufferId(id) {}

BufferInterface::~BufferInterface() { glDeleteBuffers(1, &bufferId); }

void BufferInterface::bind() const {
  glBindBuffer(bufferDescriptor.type, bufferId);
}

void BufferInterface::registerAttribute(const std::string &name, GLint location) const {
  if (bufferDescriptor.attributes.find(name) ==
      bufferDescriptor.attributes.end())
    return;
  const BufferDescriptor::Attribute &va =
      bufferDescriptor.attributes.find(name)->second;
  glVertexAttribPointer(static_cast<GLuint>(location), va.size, va.type,
                        GL_FALSE,
                        bufferDescriptor.elementSize * sizeof(float),
                        reinterpret_cast<const void *>(va.offset));
}

void BufferInterface::locateAttributes(const ShaderProgram &s, uint d) const {
  for (auto &attribute : bufferDescriptor.attributes) {
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
                            bufferDescriptor.elementSize * sizeof(float),
                            (void *) (attribute.second.offset +
                                i * componentSize * sizeof(float)));
      CHECK_GL_ERRORS;
      glVertexAttribDivisor(static_cast<GLuint>(loc + i), d);
    }
  }
}

GLuint BufferInterface::id() const { return bufferId; }

} // namespace circe
