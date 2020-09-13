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

#include <circe/gl/io/render_texture.h>
#include <circe/gl/utils/open_gl.h>

namespace circe::gl {

RenderTexture::RenderTexture(const TextureAttributes &a,
                             const TextureParameters &p)
    : Texture(a, p) /*attributes(a), parameters(p)*/ {
  /*ASSERT(a.target == p.target);

  glGenTextures(1, &textureObject);
  glBindTexture(p.target, textureObject);
  parameters.apply();
  if (a.target == GL_TEXTURE_3D)
    glTexImage3D(GL_TEXTURE_3D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, attributes.depth, 0, attributes.format,
                 attributes.type, NULL);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, 0, attributes.format, attributes.type, 0);
  CHECK_GL_ERRORS;
  glBindTexture(attributes.target, 0);
*/
  framebuffer.reset(
      new Framebuffer(attributes.width, attributes.height, attributes.depth));
  framebuffer->attachColorBuffer(textureObject, attributes.target);
  framebuffer->disable();
}

RenderTexture::~RenderTexture() {
  if (textureObject)
    glDeleteTextures(1, &textureObject);
}

void RenderTexture::render(std::function<void()> f) {
  glViewport(0, 0, static_cast<GLsizei>(attributes.width),
             static_cast<GLsizei>(attributes.height));
  framebuffer->enable();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  f();
  framebuffer->disable();
}

std::ostream &operator<<(std::ostream &out, RenderTexture &pt) {
  int width = pt.attributes.width;
  int height = pt.attributes.height;

  unsigned char *data = NULL;

  data = new unsigned char[(int)(width * height)];

  for (int i(0); i < width; ++i) {
    for (int j(0); j < height; ++j) {
      data[j * width + i] = 0;
    }
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(pt.attributes.target, pt.textureObject);
  glGetTexImage(pt.attributes.target, 0, pt.attributes.format,
                pt.attributes.type, data);

  CHECK_GL_ERRORS;

  out << width << " " << height << std::endl;

  for (int j(height - 1); j >= 0; --j) {
    for (int i(0); i < width; ++i) {
      out << (int)data[(int)(j * width + i)] << ",";
    }
    out << std::endl;
  }
  return out;
}

void RenderTexture::set(const TextureAttributes &a, const TextureParameters &p) {
  Texture::set(a, p);
  framebuffer.reset(
      new Framebuffer(attributes.width, attributes.height, attributes.depth));
  framebuffer->attachColorBuffer(textureObject, attributes.target);
  framebuffer->disable();
}

} // namespace circe
