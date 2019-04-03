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

#ifndef CIRCE_GRAPHICS_SHADER_H
#define CIRCE_GRAPHICS_SHADER_H

#include <circe/graphics/shader_manager.h>
#include <circe/io/buffer.h>

#include <ponos/ponos.h>

#include <initializer_list>
#include <set>

namespace circe {

/// Holds a program id and serves as an interface for setting its uniforms.
class ShaderProgram {
public:
  explicit ShaderProgram(int id = 0);
  ShaderProgram(const ShaderProgram &other);
  ShaderProgram(const ShaderProgram &&other);
  ShaderProgram &operator=(const ShaderProgram &other);

  /// It expects only one file of each type with extensions .fs, .vs and .gs.
  /// \brief Creates a shader program from strings.
  /// \param vs vertex shader
  /// \param gs geometry shader
  /// \param fs fragment shader
  ShaderProgram(const char *vs, const char *gs, const char *fs);
  /// It expects only one file of each type with extensions .fs, .vs and .gs.
  explicit ShaderProgram(std::initializer_list<const char *> files);
  /// \brief Creates a shader program from shader files.
  /// It expects only one file of each type with extensions .fs, .vs and .gs.
  /// \return program id. **-1** if error.
  bool loadFromFiles(std::initializer_list<const char *> files);
  /// Activate shader program
  bool begin();
  /// Deactivate shader program
  void end();
  /// \param name
  /// \param location
  void addVertexAttribute(const char *name, GLint location);
  ///
  /// \param name
  /// \param location
  void addUniform(const std::string &name, GLint location);
  /// locates atribute **name** in shader's program
  /// \param name attibute's name, must match name in shader code
  /// \return attributes layout location
  int locateAttribute(const std::string &name) const;
  /// Register shader attributes into vertex buffer
  /// \param b buffer pointer (must match attribute names)
  void registerVertexAttributes(const VertexBuffer *b);
  // Uniforms
  void setUniform(const char *name, const ponos::Transform &t);
  void setUniform(const char *name, const ponos::mat4 &m);
  void setUniform(const char *name, const ponos::mat3 &m);
  void setUniform(const char *name, const ponos::vec4 &v);
  void setUniform(const char *name, const ponos::vec3 &v);
  void setUniform(const char *name, const ponos::vec2 &v);
  void setUniform(const char *name, const ponos::point3 &v);
  void setUniform(const char *name, int i);
  void setUniform(const char *name, float f);

  bool running = false;

  friend std::ostream &operator<<(std::ostream &o, ShaderProgram shader) {
    o << "SHADER (programId " << shader.programId << ")\n";
    o << "vertex attributes:\n";
    for (auto a : shader.attrLocations)
      o << "\t" << a.first << "\t" << a.second << std::endl;
    o << "uniform list:\n";
    for (auto a : shader.uniformLocations)
      o << "\t" << a.first << "\t" << a.second << std::endl;
    o << std::endl;
    return o;
  }

protected:
  GLuint programId = 0;

  //  std::set<const char *> vertexAttributes;
  std::map<std::string, GLint> attrLocations;
  std::map<std::string, GLint> uniformLocations;

  GLint getUniLoc(const GLchar *name);
};

typedef std::shared_ptr<ShaderProgram> ShaderProgramPtr;

template <typename... TArg>
ShaderProgramPtr createShaderProgramPtr(TArg &&... Args) {
  return std::make_shared<ShaderProgram>(std::forward<TArg>(Args)...);
}

} // namespace circe

#endif // CIRCE_GRAPHICS_SHADER_H
