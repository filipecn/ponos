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

#ifndef AERGIA_GRAPHICS_SHADER_H
#define AERGIA_GRAPHICS_SHADER_H

#include "io/buffer.h"
#include "graphics/shader_manager.h"

#include <ponos.h>

#include <set>

namespace aergia {

/** \brief shader class
 * Holds a program id and serves as an interface for setting its uniforms.
 */
class Shader {
public:
  Shader(GLuint id = 0);
  /** \brief Creates a shader program from shader files.
   * It expects only one file of each type with extensions .fs, .vs and .gs.
   * \return program id. **-1** if error.
   */
  bool loadFromFiles(const char *fl...);
  /** \brief Acctivate shader program
   * \param b buffer pointer (must match attribute names)
   */
  bool begin(const VertexBuffer *b = nullptr);
  /** \brief Deactivate shader program
   */
  void end();
  /** \brief
   * \param name
   */
  void addVertexAttribute(const char *name);
  // Uniforms
  void setUniform(const char *name, const ponos::mat4 &m);
  void setUniform(const char *name, const ponos::mat3 &m);
  void setUniform(const char *name, const ponos::vec4 &v);
  void setUniform(const char *name, const ponos::vec3 &v);
  void setUniform(const char *name, const ponos::vec2 &v);
  void setUniform(const char *name, int i);
  void setUniform(const char *name, float f);

  bool running;

protected:
  GLuint programId;

  std::set<const char *> vertexAttributes;

  GLint getUniLoc(const GLchar *name);
};

} // aergia namespace

#endif // AERGIA_GRAPHICS_SHADER_H
