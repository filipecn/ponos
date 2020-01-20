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

#include <circe/graphics/shader.h>

namespace circe {

ShaderProgram::ShaderProgram(const ShaderProgram &other) {
  programId = other.programId;
  for (const auto& a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto& a : other.uniformLocations)
    uniformLocations[a.first] = a.second;
}

ShaderProgram::ShaderProgram(const ShaderProgram &&other) noexcept {
  programId = other.programId;
  for (const auto& a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto& a : other.uniformLocations)
    uniformLocations[a.first] = a.second;
}

ShaderProgram &ShaderProgram::operator=(const ShaderProgram &other) {
  programId = other.programId;
  for (const auto& a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto& a : other.uniformLocations)
    uniformLocations[a.first] = a.second;
  return *this;
}

ShaderProgram::ShaderProgram(int id) {
  FATAL_ASSERT(id >= 0);
  programId = static_cast<GLuint>(id);
}

ShaderProgram::ShaderProgram(const char *vs, const char *gs, const char *fs)
    : ShaderProgram(ShaderManager::loadFromTexts(vs, gs, fs)) {}

ShaderProgram::ShaderProgram(std::initializer_list<const char *> files)
    : programId(0) {
  loadFromFiles(files);
}

bool ShaderProgram::loadFromFiles(std::initializer_list<const char *> files) {
  int program = ShaderManager::loadFromFiles(files);
  if (program < 0)
    return false;
  programId = static_cast<GLuint>(program);
  return true;
}

bool ShaderProgram::begin() {
  if (!ShaderManager::instance().useShader(programId))
    return false;
  return true;
}

void ShaderProgram::registerVertexAttributes(const VertexBuffer *b) {
  if (!ShaderManager::instance().useShader(programId))
    return;
  for (const auto& va : attrLocations) {
    GLint attribute = va.second;
    glEnableVertexAttribArray(attribute);
    CHECK_GL_ERRORS;
    b->registerAttribute(va.first, attribute);
    CHECK_GL_ERRORS;
  }
  end();
}

int ShaderProgram::locateAttribute(const std::string &name) const {
  auto it = attrLocations.find(name);
  if (it == attrLocations.end())
    return -1;
  return it->second;
  //  return glGetAttribLocation(programId, name.c_str());
}

void ShaderProgram::end() { glUseProgram(0); }

void ShaderProgram::addVertexAttribute(const char *name, GLint location) {
  //  vertexAttributes.insert(name);
  attrLocations[name] = location;
}

void ShaderProgram::addUniform(const std::string &name, GLint location) {
  uniformLocations[name] = location;
}

void ShaderProgram::setUniform(const char *name, const ponos::Transform &t) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Must be added first.)\n";
    return;
  }
  glUniformMatrix4fv(loc, 1, GL_FALSE, &t.matrix().m[0][0]);
}

void ShaderProgram::setUniform(const char *name, const ponos::mat4 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void ShaderProgram::setUniform(const char *name, const ponos::mat3 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniformMatrix3fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void ShaderProgram::setUniform(const char *name, const ponos::vec4 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform4fv(loc, 1, &v.x);
}

void ShaderProgram::setUniform(const char *name, const ponos::vec3 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform3fv(loc, 1, &v.x);
}

void ShaderProgram::setUniform(const char *name, const ponos::point3 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  CHECK_GL(glUniform3fv(loc, 1, &v.x));
}

void ShaderProgram::setUniform(const char *name, const ponos::vec2 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform2fv(loc, 1, &v.x);
}

void ShaderProgram::setUniform(const char *name, const Color &c) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform4fv(loc, 1, &c.r);
}

void ShaderProgram::setUniform(const char *name, int i) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform1i(loc, i);
}

void ShaderProgram::setUniform(const char *name, float f) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform1f(loc, f);
}
GLint ShaderProgram::getUniLoc(const GLchar *name) {
  //  if (!ShaderManager::instance().useShader(programId))
  //    return -1;
  auto it = uniformLocations.find(name);
  if (it == uniformLocations.end())
    return -1;
  return it->second;
  //  return glGetUniformLocation(programId, name);
}

} // namespace circe
