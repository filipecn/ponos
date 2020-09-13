/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file shadow_map.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-08-09
///
///\brief

#include <circe/gl/graphics/shadow_map.h>

namespace circe::gl {

ShadowMap::ShadowMap(const ponos::size2 &size) : size_(size) {
  // setup shader program
  Shader vertex_shader("#version 430 core\n"
                       "layout (location = 0) in vec3 position;\n"
                       "layout (location = 1) uniform mat4 lightSpaceMatrix;\n"
                       "layout (location = 2) uniform mat4 model;\n"
                       "void main()\n"
                       "{ gl_Position = lightSpaceMatrix * model * vec4(position, 1.0); }", GL_VERTEX_SHADER);
  Shader fragment_shader("#version 430 core\nvoid main(){}\n", GL_FRAGMENT_SHADER);
  program_.attach(vertex_shader);
  program_.attach(fragment_shader);
  if(!program_.link()) {
    std::cerr << program_.err << std::endl;
    exit(-1);
  }
  // setup depth texture attributes and parameters
  TextureAttributes attributes;
  attributes.height = size.height;
  attributes.width = size.width;
  attributes.depth = 1;
  attributes.target = GL_TEXTURE_2D;
  attributes.internalFormat = GL_DEPTH_COMPONENT;
  attributes.format = GL_DEPTH_COMPONENT;
  attributes.type = GL_FLOAT;
  TextureParameters parameters(GL_TEXTURE_2D);
  parameters[GL_TEXTURE_MIN_FILTER] = GL_NEAREST;
  parameters[GL_TEXTURE_MAG_FILTER] = GL_NEAREST;
  parameters[GL_TEXTURE_WRAP_S] = GL_REPEAT;
  parameters[GL_TEXTURE_WRAP_T] = GL_REPEAT;
  // set depth texture
  depth_map_.set(attributes, parameters);
  // set buffer
  depth_buffer_.set(size.width, size.height);
  depth_buffer_.attachColorBuffer(depth_map_.textureObjectId(), depth_map_.target(), GL_DEPTH_ATTACHMENT);
  depth_buffer_.enable();
  // Since we don't need a color buffer and Open GL expects one, we must tell Open GL:
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  depth_buffer_.disable();
  CHECK_GL_ERRORS;
}

ShadowMap::~ShadowMap() {

}

void ShadowMap::render(std::function<void()> f) {
  glViewport(0, 0, size_.width, size_.height);
  depth_buffer_.enable();
  glClear(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  program_.use();
  program_.setUniform("lightSpaceMatrix", ponos::transpose(light_transform_.matrix()));
  program_.setUniform("model", ponos::Transform().matrix());
  if (f)
    f();
  depth_buffer_.disable();
}

void ShadowMap::bind() const {
  glBindTexture(GL_TEXTURE_2D, depth_map_.textureObjectId());
}

void ShadowMap::setLight(const Light &light) {
  ponos::Transform projection, view;
  if (light.type == circe::LightTypes::DIRECTIONAL) {
    projection = ponos::ortho(-10, 10, -10, 10, 1, 7.5);
    view = ponos::lookAtRH(ponos::point3() - 10.f * light.direction, ponos::point3(), ponos::vec3(0, 1, 0));
  }
  light_transform_ = projection * view;
}

const ponos::Transform &ShadowMap::light_transform() const { return light_transform_; }

const Texture& ShadowMap::depthMap() const {
  return depth_map_;
}

}