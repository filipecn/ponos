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

#include <aergia/io/buffer.h>

#include <aergia/graphics/shader.h>

namespace aergia {

void BufferInterface::locateAttributes(const Shader &s, uint d) const {
  for(auto& attribute : bufferDescriptor.attributes) {
    GLint loc = s.locateAttribute(attribute.first);
    if(loc < 0)
      continue;
    glEnableVertexAttribArray(static_cast<GLuint>(loc));
    glVertexAttribPointer(static_cast<GLuint>(loc), attribute.second.size,
                          attribute.second.type, GL_FALSE,
                          bufferDescriptor.elementSize * sizeof(float),
                          (void *)(attribute.second.offset));
    CHECK_GL_ERRORS;
    glVertexAttribDivisor(static_cast<GLuint>(loc), d);
  }
}

} // aergia namespace