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

#include <aergia/graphics/shader.h>

namespace aergia {

Shader::Shader(GLuint id) : programId(id) { running = false; }

bool Shader::loadFromFiles(const char *fl...) {
  running = false;
  int program = ShaderManager::instance().loadFromFiles(fl);
  if (program < 0)
    return false;
  programId = static_cast<GLuint>(program);
  return true;
}

bool Shader::begin(const VertexBuffer *b) {
  if (running)
    return true;
  if (!ShaderManager::instance().useShader(programId))
    return false;
  if (b) {
    for (auto va : vertexAttributes) {
      GLint attribute = glGetAttribLocation(programId, va);
      glEnableVertexAttribArray(attribute);
      b->registerAttribute(std::string(va), attribute);
    }
  }
  running = true;
  return true;
}

void Shader::end() {
  glUseProgram(0);
  running = false;
}

void Shader::addVertexAttribute(const char *name) {
  vertexAttributes.insert(name);
}

void Shader::setUniform(const char *name, const ponos::mat4 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void Shader::setUniform(const char *name, const ponos::mat3 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniformMatrix3fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void Shader::setUniform(const char *name, const ponos::vec4 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniform4fv(loc, 1, &v.x);
}

void Shader::setUniform(const char *name, const ponos::vec3 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniform3fv(loc, 1, &v.x);
}

void Shader::setUniform(const char *name, const ponos::vec2 &v) {
  bool wasNotRunning = !running;
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniform2fv(loc, 1, &v.x);
  if (wasNotRunning)
    end();
}

void Shader::setUniform(const char *name, int i) {
  bool wasNotRunning = !running;
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniform1i(loc, i);
  if (wasNotRunning)
    end();
}

void Shader::setUniform(const char *name, float f) {
  bool wasNotRunning = !running;
  GLint loc = getUniLoc(name);
  if (loc == -1)
    return;
  glUniform1f(loc, f);
  if (wasNotRunning)
    end();
}

GLint Shader::getUniLoc(const GLchar *name) {
  if (!ShaderManager::instance().useShader(programId))
    return -1;
  return glGetUniformLocation(programId, name);
}

} // aergia namespace
