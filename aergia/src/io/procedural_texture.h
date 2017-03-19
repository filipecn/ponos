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

#ifndef AERGIA_IO_PROCEDURAL_TEXTURE_H
#define AERGIA_IO_PROCEDURAL_TEXTURE_H

#include "io/framebuffer.h"
#include "io/texture_parameters.h"

#include <ponos.h>
#include <functional>
#include <memory>

namespace aergia {

/** \brief Renders the image into a texture directly from the framebuffer.
 */
class ProceduralTexture {
public:
  /** \brief Constructor.
   * \param a texture attributes
   * \param p texture parameters
   */
  ProceduralTexture(const TextureAttributes &a, const TextureParameters &p);
  virtual ~ProceduralTexture();
  void bind(GLenum t);
  void render(std::function<void()> f);
  friend std::ostream &operator<<(std::ostream &out, ProceduralTexture &pt);

protected:
  TextureAttributes attributes;
  TextureParameters parameters;
  std::shared_ptr<Framebuffer> framebuffer;
  GLuint textureObject;
};

} // aergia namespace

#endif // AERGIA_IO_PROCEDURAL_TEXTURE_H
