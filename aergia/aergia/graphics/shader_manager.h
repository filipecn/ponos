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

#ifndef AERGIA_GRAPHICS_SHADER_MANAGER_H
#define AERGIA_GRAPHICS_SHADER_MANAGER_H

#include <aergia/utils/open_gl.h>

#include <map>
#include <stdarg.h>
#include <string>
#include <vector>

namespace aergia {

/// \brief singleton
/// Manages shader programs
class ShaderManager {
 public:
  static ShaderManager &instance() { return instance_; }
  virtual ~ShaderManager() = default;
  /// \brief Creates a shader program from shader files.
  /// It expects only one file of each type with extensions .fs, .vs and .gs.
  ///\return program id. **-1** if error.
  int loadFromFiles(const char *fl, ...);
  /// \brief Creates a shader program from strings.
  /// \param vs vertex shader
  /// \param gs geometry shader
  /// \param fs fragment shader
  /// \return program id. **-1** if error.
  int loadFromTexts(const char *vs, const char *gs, const char *fs);
  int loadFromText(const char *s, GLuint shaderType);
  /// \brief use program
  ///\param program **[in]** program's id
  ///Activate program
  ///\return **true** if success
  bool useShader(GLuint program);

  ShaderManager(ShaderManager const &) = delete;
  void operator=(ShaderManager const &) = delete;
 private:
  ShaderManager();

  GLuint createProgram(const GLchar *, const GLchar *);
  GLuint compile(const char *shaderSource, GLuint shaderType);
  GLuint createProgram(const std::vector<GLuint> &objects);

  static ShaderManager instance_;
};

#define AERGIA_NO_VAO_VS "#version 440 core\n out vec2 texCoord;" \
    "void main() {" \
"    float x = -1.0 + float((gl_VertexID & 1) << 2);"\
"    float y = -1.0 + float((gl_VertexID & 2) << 1);"\
"    texCoord.x = (x+1.0)*0.5;"\
"    texCoord.y = (y+1.0)*0.5;"\
"    gl_Position = vec4(x, y, 0, 1);"\
"}"

#define AERGIA_NO_VAO_FS \
"#version 440 core\n" \
"out vec4 outColor;" \
"in vec2 texCoord;"\
"uniform sampler2D tex;"\
"void main() {"\
"outColor = texture(tex, texCoord);"\
"}"

#define AERGIA_INSTANCES_VS \
"#version 440 core\n" \
"layout (location = 0) in vec3 position;" \
"layout (location = 1) in vec3 pos;"  \
"layout (location = 2) in float scale;" \
"layout (location = 3) in vec4 col;"  \
"layout (location = 4) uniform mat4 view_matrix;" \
"layout (location = 5) uniform mat4 projection_matrix;" \
"out VERTEX {" \
"vec4 color;" \
"} vertex;" \
"void main() {" \
"    mat4 model_matrix;" \
"    model_matrix[0] = vec4(scale, 0, 0, 0);" \
"    model_matrix[1] = vec4(0, scale, 0, 0);" \
"    model_matrix[2] = vec4(0, 0, scale, 0);" \
"    model_matrix[3] = vec4(pos.x, pos.y, pos.z, 1);" \
"    mat4 model_view_matrix = view_matrix * model_matrix;\n" \
"    gl_Position = projection_matrix * model_view_matrix * " \
"vec4(position,1);" \
"   vertex.color = col;" \
"}"

#define AERGIA_INSTANCES_FS \
"#version 440 core\n" \
    "in VERTEX { vec4 color; } vertex;" \
    "out vec4 outColor;" \
    "void main() {" \
    "   outColor = vertex.color;" \
    "}";

} // aergia namespace

#endif // AERGIA_GRAPHICS_SHADER_MANAGER_H
