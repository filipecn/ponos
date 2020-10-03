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

#include <circe/gl/io/texture.h>
#include <circe/gl/utils/open_gl.h>

namespace circe::gl {

Texture::Texture() = default;

Texture::Texture(const TextureAttributes &a, const TextureParameters &p)
    : attributes(a), parameters(p) {
  set(a, p);
}

Texture::~Texture() {
  if (textureObject)
    glDeleteTextures(1, &textureObject);
}

void Texture::set(const TextureAttributes &a, const TextureParameters &p) {
  ASSERT(a.target == p.target);
  attributes = a;
  parameters = p;
  glGenTextures(1, &textureObject);
  ASSERT(textureObject);
  glBindTexture(p.target, textureObject);
  parameters.apply();
  if (a.target == GL_TEXTURE_3D)
    glTexImage3D(GL_TEXTURE_3D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, attributes.depth, 0, attributes.format,
                 attributes.type, attributes.data);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, 0, attributes.format, attributes.type,
                 attributes.data);
  CHECK_GL_ERRORS;
  glBindTexture(attributes.target, 0);
}

void Texture::setTexels(unsigned char *texels) {
  attributes.data = texels;
  glBindTexture(parameters.target, textureObject);
  if (attributes.target == GL_TEXTURE_3D)
    glTexImage3D(GL_TEXTURE_3D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, attributes.depth, 0, attributes.format,
                 attributes.type, attributes.data);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, attributes.internalFormat, attributes.width,
                 attributes.height, 0, attributes.format, attributes.type,
                 attributes.data);
  CHECK_GL_ERRORS;
  glBindTexture(attributes.target, 0);
}

void Texture::bind(GLenum t) const {
  glActiveTexture(t);
  glBindTexture(attributes.target, textureObject);
}

void Texture::bindImage(GLenum t) const {
  glActiveTexture(t);
  glBindImageTexture(0, textureObject, 0, GL_FALSE, 0, GL_WRITE_ONLY,
                     attributes.internalFormat);
  CHECK_GL_ERRORS;
}

ponos::size3 Texture::size() const {
  return ponos::size3(attributes.width, attributes.height, attributes.depth);
}

GLuint Texture::textureObjectId() const { return textureObject; }

GLenum Texture::target() const { return parameters.target; }

std::ostream &operator<<(std::ostream &out, Texture &pt) {
  auto width = static_cast<int>(pt.attributes.width);
  auto height = static_cast<int>(pt.attributes.height);

  u8 *data = nullptr;
  auto bytes_per_texel = OpenGL::dataSizeInBytes(pt.attributes.type);
  auto memory_size = width * height * OpenGL::dataSizeInBytes(pt.attributes.type);
  data = new u8[memory_size];
  memset(data, 0, memory_size);

  glActiveTexture(GL_TEXTURE0);
  std::cerr << "texture object " << pt.textureObject << std::endl;
  out << width << " x " << height << " texels of type " <<
      OpenGL::TypeToStr(pt.attributes.type) << " (" << bytes_per_texel
      << " bytes per texel)" << std::endl;
  out << "total memory: " << memory_size << " bytes\n";
  glBindTexture(pt.attributes.target, pt.textureObject);
  glGetTexImage(pt.attributes.target, 0, pt.attributes.format,
                pt.attributes.type, data);

  CHECK_GL_ERRORS;

  for (int j(height - 1); j >= 0; --j) {
    for (int i(0); i < width; ++i) {
      switch (pt.attributes.type) {
      case GL_FLOAT:out << reinterpret_cast<f32 *>(data)[(int) (j * width + i)] << ",";
        break;
      default:out << static_cast<int>(data[(int) (j * width + i)]) << ",";
      }
    }
    out << std::endl;
  }

  delete[] data;

  return out;
}

std::vector<unsigned char> Texture::texels() const {
  auto width = static_cast<int>(attributes.width);
  auto height = static_cast<int>(attributes.height);

  std::vector<unsigned char> data(4 * width * height, 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(attributes.target, textureObject);
  glGetTexImage(attributes.target, 0, attributes.format, attributes.type,
                &data[0]);
  CHECK_GL_ERRORS;
  return data;
}

} // namespace circe
