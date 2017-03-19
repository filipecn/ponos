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

#include "io/framebuffer.h"

namespace aergia {

Framebuffer::Framebuffer() {
  width = height = depth = 0;
  framebufferObject = renderBufferObject = 0;
}

Framebuffer::Framebuffer(size_t w, size_t h, size_t d)
    : width(w), height(h), depth(d) {
  set(width, height, depth);
}

Framebuffer::~Framebuffer() {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  if (framebufferObject)
    glDeleteFramebuffers(1, &framebufferObject);
}

void Framebuffer::set(size_t w, size_t h, size_t d) {
  glGenFramebuffers(1, &framebufferObject);
  glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);

  glGenRenderbuffers(1, &renderBufferObject);
  glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);

  // attach the renderbuffer to depth attachment point
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, // 1. fbo target: GL_FRAMEBUFFER
                            GL_DEPTH_ATTACHMENT, // 2. attachment point
                            GL_RENDERBUFFER, // 3. rbo target: GL_RENDERBUFFER
                            renderBufferObject); // 4. rbo ID

  CHECK_GL_ERRORS;
  CHECK_FRAMEBUFFER;

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::attachColorBuffer(GLuint textureId, GLenum target,
                                    GLenum attachmentPoint) {
  glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);

  // attach the texture to FBO color attachment point
  glFramebufferTexture2D(GL_FRAMEBUFFER,  // 1. fbo target: GL_FRAMEBUFFER
                         attachmentPoint, // 2. attachment point
                         target,          // 3. tex target: GL_TEXTURE_2D
                         textureId,       // 4. tex ID
                         0);              // 5. mipmap level: 0(base)
  CHECK_GL_ERRORS;
  CHECK_FRAMEBUFFER;

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::enable() {
  glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
  glBindRenderbuffer(GL_RENDERBUFFER, renderBufferObject);
}

void Framebuffer::disable() {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

} // aergia namespace
