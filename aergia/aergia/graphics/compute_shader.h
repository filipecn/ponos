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

#ifndef AERGIA_GRAPHICS_COMPUTE_SHADER_H
#define AERGIA_GRAPHICS_COMPUTE_SHADER_H

#include <aergia/utils/open_gl.h>
#include <aergia/io/texture.h>
#include <aergia/graphics/shader.h>

namespace aergia {

class ComputeShader {
public:
  ComputeShader(const TextureAttributes &a, const TextureParameters &p, const char* source);
  virtual ~ComputeShader();
  bool compute();
  void bindTexture(GLenum t) const;

private:
  GLuint programId;
  std::unique_ptr<Texture> texture;
  ponos::uivec3 groupSize;
};

} // aergia namespace

#endif //AERGIA_GRAPHICS_COMPUTE_SHADER_H
