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

#include <circe/gl/graphics/shader.h>
#include <ponos/common/file_system.h>

namespace circe::gl {

Shader::Shader() = default;

Shader::Shader(GLuint type) : type_(type) {}

Shader::Shader(const std::string &code, GLuint type) : Shader(type) {
  compile(code, type);
}

Shader::Shader(const ponos::Path &code, GLuint type) : Shader(type) {
  compile(code, type);
}

Shader::Shader(Shader &other) : type_(other.type_) {
  glDeleteShader(id_);
  id_ = other.id_;
  other.id_ = 0;
}

Shader::Shader(Shader &&other) noexcept: Shader(other) {}

Shader::~Shader() {
  glDeleteShader(id_);
}

void Shader::setType(GLuint type) {
  type_ = type;
  glDeleteShader(id_);
  id_ = glCreateShader(type_);
}

GLuint Shader::type() const { return type_; }

bool Shader::compile(const std::string &code) {
  err.clear();

  if (!id_)
    id_ = glCreateShader(type_);
  CHECK_GL_ERRORS;

  GLint compiled;
  const char *source_code = code.c_str();

  glShaderSource(id_, 1, &source_code, nullptr);
  glCompileShader(id_);

  glGetShaderiv(id_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    // retrieve error string
    GLsizei infolog_length = 0;
    int chars_written = 0;
    glGetShaderiv(id_, GL_INFO_LOG_LENGTH, &infolog_length);
    if (infolog_length > 0) {
      GLchar *log = (GLchar *) malloc((size_t) infolog_length);
      glGetShaderInfoLog(id_, infolog_length, &chars_written, log);
      err = log;
      free(log);
    }
    return false;
  }
  return true;
}

bool Shader::compile(const std::string &code, GLuint type) {
  setType(type);
  return compile(code);
}

bool Shader::compile(const ponos::Path &file, GLuint type) {
  setType(type);
  return compile(file.read());
}

GLuint Shader::id() const {
  return id_;
}

ShaderProgram::ShaderProgram(const ShaderProgram &other) {
  programId = other.programId;
  for (const auto &a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto &a : other.uniformLocations)
    uniformLocations[a.first] = a.second;
}

ShaderProgram::ShaderProgram(const ShaderProgram &&other) noexcept {
  programId = other.programId;
  for (const auto &a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto &a : other.uniformLocations)
    uniformLocations[a.first] = a.second;
}

ShaderProgram &ShaderProgram::operator=(const ShaderProgram &other) {
  programId = other.programId;
  for (const auto &a : other.attrLocations)
    attrLocations[a.first] = a.second;
  for (const auto &a : other.uniformLocations)
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

[[maybe_unused]] void ShaderProgram::registerVertexAttributes(const GLVertexBuffer *b) {
  if (!ShaderManager::instance().useShader(programId))
    return;
  for (const auto &va : attrLocations) {
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

Program::Program() = default;

Program::Program(const std::vector<ponos::Path> &files) : Program() {
  std::vector<Shader> shaders;
  for (const auto &file : files) {
    GLuint type = GL_VERTEX_SHADER;
    if (file.extension() == "frag")
      type = GL_FRAGMENT_SHADER;
    shaders.emplace_back(file, type);
  }
  // check for shader errors
  for (const auto &shader : shaders)
    if (!shader.id() || !shader.err.empty()) {
      err = shader.err;
      return;
    }
  link(shaders);
}

Program::Program(std::initializer_list<ponos::Path> files) : Program() {
  link(files);
}

Program::Program(const std::vector<Shader> &shaders) : Program() {
  link(shaders);
}

Program::Program(Program &other) : id_(other.id_), attr_locations_(std::move(other.attr_locations_)),
                                   uniform_locations_(std::move(other.uniform_locations_)) {
  other.id_ = 0;
}

Program::Program(Program &&other) noexcept: Program(other) {}

Program::~Program() {
  destroy();
}

void Program::destroy() {
  glDeleteProgram(id_);
  id_ = 0;
}

void Program::attach(const Shader &shader) {
  if (!id_)
    create();
  linked_ = false;
  glAttachShader(id_, shader.id_);
  CHECK_GL_ERRORS;
}

void Program::attach(const std::vector<Shader> &shader_list) {
  for (const auto &s : shader_list)
    attach(s);
}

bool Program::link() {
  if (!id_)
    create();
  glLinkProgram(id_);
  if (!checkLinkageErrors())
    return false;
  cacheLocations();
  return true;
}

bool Program::link(const std::vector<Shader> &shaders) {
  if (!id_)
    create();
  for (const auto &shader : shaders)
    glAttachShader(id_, shader.id_);
  return link();
}

bool Program::link(const std::vector<ponos::Path> &shader_file_list) {
  std::vector<Shader> shaders;
  for (const auto &file : shader_file_list) {
    GLuint type = GL_VERTEX_SHADER;
    if (file.extension() == "frag")
      type = GL_FRAGMENT_SHADER;
    shaders.emplace_back(file, type);
  }
  // check for shader errors
  for (const auto &shader : shaders)
    if (!shader.id() || !shader.err.empty()) {
      err = shader.err;
      return false;
    }
  return link(shaders);
}

bool Program::use() {
  if (!id_)
    create();
  if (!linked_)
    if (!link())
      return false;
  CHECK_GL(glUseProgram(id_));
  return true;
}

void Program::cacheLocations() {
  attr_locations_.clear();
  uniform_locations_.clear();

  GLint i;
  GLint count;

  GLint size; // size of the variable
  GLenum type; // type of the variable (float, vec3 or mat4, etc)

  const GLsizei bufSize = 30; // maximum name length
  GLchar name[bufSize]; // variable name in GLSL
  GLsizei length; // name length

  glGetProgramiv(id_, GL_ACTIVE_ATTRIBUTES, &count);
  for (i = 0; i < count; i++) {
    glGetActiveAttrib(id_, (GLuint) i, bufSize, &length, &size, &type, name);
    attr_locations_[name] = glGetAttribLocation(id_, name);
  }

  glGetProgramiv(id_, GL_ACTIVE_UNIFORMS, &count);
  for (i = 0; i < count; i++) {
    glGetActiveUniform(id_, (GLuint) i, bufSize, &length, &size, &type, name);
    uniform_locations_[name] = glGetUniformLocation(id_, name);
  }
}

void Program::addVertexAttribute(const std::string &name, GLint location) {
  attr_locations_[name] = location;
}

void Program::addUniform(const std::string &name, GLint location) {
  uniform_locations_[name] = location;
}

int Program::locateAttribute(const std::string &name) const {
  auto it = attr_locations_.find(name);
  if (it == attr_locations_.end())
    return -1;
  return it->second;
}

GLuint Program::id() const { return id_; }

void Program::setUniform(const std::string &name, const ponos::Transform &t) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Must be added first.)\n";
    return;
  }
  glUniformMatrix4fv(loc, 1, GL_FALSE, &t.matrix().m[0][0]);
}

void Program::setUniform(const std::string &name, const ponos::mat4 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void Program::setUniform(const std::string &name, const ponos::mat3 &m) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniformMatrix3fv(loc, 1, GL_FALSE, &m.m[0][0]);
}

void Program::setUniform(const std::string &name, const ponos::vec4 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform4fv(loc, 1, &v.x);
}

void Program::setUniform(const std::string &name, const ponos::vec3 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform3fv(loc, 1, &v.x);
}

void Program::setUniform(const std::string &name, const ponos::point3 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  CHECK_GL(glUniform3fv(loc, 1, &v.x));
}

void Program::setUniform(const std::string &name, const ponos::vec2 &v) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform2fv(loc, 1, &v.x);
}

void Program::setUniform(const std::string &name, const Color &c) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform4fv(loc, 1, &c.r);
}

void Program::setUniform(const std::string &name, int i) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform1i(loc, i);
}

void Program::setUniform(const std::string &name, float f) {
  GLint loc = getUniLoc(name);
  if (loc == -1) {
    std::cerr << "Attribute " << name
              << " not located. (Probably has not been added.\n";
    return;
  }
  glUniform1f(loc, f);
}

GLint Program::getUniLoc(const std::string &name) {
  auto it = uniform_locations_.find(name);
  if (it == uniform_locations_.end())
    return -1;
  return it->second;
}

void Program::create() {
  id_ = glCreateProgram();
  CHECK_GL_ERRORS;
}

bool Program::checkLinkageErrors() {
  GLint linked = 0;
  glGetProgramiv(id_, GL_LINK_STATUS, &linked);
  if (!linked) {
    // retrieve error string
    GLsizei infolog_length = 0;
    int chars_written = 0;
    glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &infolog_length);
    if (infolog_length > 0) {
      GLchar *log = (GLchar *) malloc((int) (infolog_length + 1));
      glGetProgramInfoLog(id_, infolog_length, &chars_written, log);
      err = static_cast<const char *>(log);
      free(log);
    }
    linked_ = false;
    return false;
  }
  linked_ = true;
  return true;
}

std::ostream &operator<<(std::ostream &o, const Program &program) {
  GLint i;
  GLint count;

  GLint size; // size of the variable
  GLenum type; // type of the variable (float, vec3 or mat4, etc)

  const GLsizei bufSize = 30; // maximum name length
  GLchar name[bufSize]; // variable name in GLSL
  GLsizei length; // name length

  glGetProgramiv(program.id_, GL_ACTIVE_ATTRIBUTES, &count);
  o << "Active Attributes: " << count << std::endl;

  for (i = 0; i < count; i++) {
    glGetActiveAttrib(program.id_, (GLuint) i, bufSize, &length, &size, &type, name);
    o << "Attribute #" << i << " Type: " << type << " Name: " << name << " location "
      << glGetAttribLocation(program.id_, name) << std::endl;
  }

  glGetProgramiv(program.id_, GL_ACTIVE_UNIFORMS, &count);
  o << "Active Uniforms: " << count << std::endl;

  for (i = 0; i < count; i++) {
    glGetActiveUniform(program.id_, (GLuint) i, bufSize, &length, &size, &type, name);

    o << "Uniform #" << i << " Type: " << type << " Name: " << name << " location "
      << glGetUniformLocation(program.id_, name) << std::endl;
  }

  return o;
}

} // namespace circe
