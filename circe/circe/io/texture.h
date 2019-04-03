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

#ifndef CIRCE_IO_TEXTURE_H
#define CIRCE_IO_TEXTURE_H

#include <circe/io/texture_parameters.h>

namespace circe {
class Texture {
public:
  Texture();
  /// \param a texture attributes
  /// \param p texture parameters
  Texture(const TextureAttributes &a, const TextureParameters &p);
  virtual ~Texture();
  /// \param a texture attributes
  /// \param p texture parameters
  void set(const TextureAttributes &a, const TextureParameters &p);
  /// Binds texture
  /// \param t texture unit (ex: GL_TEXTURE0)
  virtual void bind(GLenum t) const;
  /// Binds image texture to an image unit for  the purpose of reading and
  /// writing it from shaders
  /// \param t texture unit  (ex: GL_TEXTURE0)
  virtual void bindImage(GLenum t) const;
  /// retrieve texture pixel data
  /// \return list of pixels by row major
  std::vector<unsigned char> texels() const;
  ponos::uivec3 size() const;
  GLuint textureObjectId() const;
  friend std::ostream &operator<<(std::ostream &out, Texture &pt);

protected:
  TextureAttributes attributes;
  TextureParameters parameters;
  GLuint textureObject;
};

} // namespace circe

#endif // CIRCE_IO_TEXTURE_H