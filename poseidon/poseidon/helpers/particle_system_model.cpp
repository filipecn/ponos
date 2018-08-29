/*
 * Copyright (c) 2018 FilipeCN
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

#include "particle_system_model.h"

namespace poseidon {

ParticleSystemModel::ParticleSystemModel(ParticleSystem &ps, float r)
    : particleRadius(r), curProperty_(-1), activeParticle_(-1), ps_(ps),
      posBuffer_(0), colorBuffer_(0) {
  // initialize instances set
  const char *vs = "#version 440 core\n"
      // regular vertex attributes
      "layout (location = 0) in vec3 position;"
      "layout (location = 1) in vec3 normal;"
      "layout (location = 2) in vec2 texcoord;"
      // per instance attributes
      "layout (location = 3) in vec3 pos;"  // instance position
      "layout (location = 4) in float rad;" // instance radius
      "layout (location = 5) in vec4 col;"  // instance color
      // constants accross draw
      "layout (location = 6) uniform mat4 view_matrix;"
      "layout (location = 7) uniform mat4 projection_matrix;"
      // output to fragment shader
      "out VERTEX {"
      "vec4 color;"
      "vec3 normal;"
      "vec2 uv;"
      "} vertex;"
      "void main() {"
      // define model_matrix: instance transform
      "    mat4 model_matrix;"
      "    model_matrix[0] = vec4(rad, 0, 0, 0);"
      "    model_matrix[1] = vec4(0, rad, 0, 0);"
      "    model_matrix[2] = vec4(0, 0, rad, 0);"
      "    model_matrix[3] = vec4(pos.x, pos.y, pos.z, 1);"
      "    mat4 model_view_matrix = view_matrix * model_matrix;\n"
      "    gl_Position = projection_matrix * model_view_matrix * "
      "vec4(position,1);"
      "    vertex.normal = normalize(model_matrix * vec4(normal, 0)).xyz;"
      "   vertex.color = col;"
      "}";
  const char *fs = "#version 440 core\n"
      "in VERTEX { vec4 color; vec3 normal; vec2 uv; } vertex;"
      "out vec4 outColor;"
      "void main() {"
      "   outColor = vertex.color;"
      "}";
  // generate base mesh
  particleMesh_.reset(ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 3, false, false));
  // create a vertex buffer for base mesh
  particleSceneMesh_.reset(new aergia::SceneMesh(*particleMesh_.get()));
  aergia::ShaderProgram shader(vs, nullptr, fs);
  shader.addVertexAttribute("position", 0);
  shader.addVertexAttribute("normal", 1);
  shader.addVertexAttribute("texcoord", 2);
  shader.addVertexAttribute("pos", 3);
  shader.addVertexAttribute("rad", 4);
  shader.addVertexAttribute("col", 5);
  shader.addUniform("view_matrix", 6);
  shader.addUniform("projection_matrix", 7);
  instances_.reset(
      new aergia::InstanceSet(*particleSceneMesh_.get(), shader, ps_.size()));
  // create a buffer for particles positions + sizes
  aergia::BufferDescriptor posSiz;
  posSiz.elementSize = 4;  // x y z s
  posSiz.dataType = GL_FLOAT;
  posSiz.type = GL_ARRAY_BUFFER;
  posSiz.use = GL_STREAM_DRAW;
  posSiz.addAttribute("pos", 3 /* x y z*/, 0, posSiz.dataType);
  posSiz.addAttribute("rad", 1 /* s */, 3 * sizeof(float), posSiz.dataType);
  posBuffer_ = instances_->add(posSiz);
  // create a buffer for particles colors
  aergia::BufferDescriptor col;
  col.elementSize = 4;  // r g b a
  col.dataType = GL_FLOAT;
  col.type = GL_ARRAY_BUFFER;
  col.use = GL_STREAM_DRAW;
  col.addAttribute("col", 4 /* r g b a */, 0, col.dataType);
  colorBuffer_ = instances_->add(col);
  update();
}

ParticleSystemModel::~ParticleSystemModel() = default;

void ParticleSystemModel::draw(const aergia::CameraInterface* camera, ponos::Transform t) {
  instances_->draw(camera, t);
}

bool ParticleSystemModel::intersect(const ponos::Ray3 &r, float *t) {
  return SceneObject::intersect(r, t);
}

void ParticleSystemModel::update() {
  double minValue = 0., maxValue = 0.;
  bool usingPalette = false;
  if (curProperty_ >= 0 && paletteD_.find(static_cast<size_t>(curProperty_)) != paletteD_.end()) {
    minValue = ps_.maxValue<double>(static_cast<size_t>(curProperty_));
    maxValue = ps_.minValue<double>(static_cast<size_t>(curProperty_));
    usingPalette = true;
  }
  // update positions
  ps_.iterateParticles([&](ParticleSystem::ParticleAccessor acc) {
    auto p = instances_->instanceF(posBuffer_, acc.id());
    auto pos = acc.position();
    p[0] = pos[0];
    p[1] = pos[1];
    p[2] = pos[2];
    p[3] = particleRadius;
    aergia::Color color = particleColor;
    if (usingPalette) {
      auto t = static_cast<float>(ponos::smoothStep(acc.property<double>(static_cast<uint>(curProperty_)),
                                                    minValue,
                                                    maxValue));
      color = paletteD_[curProperty_](t, particleColor.a);
    }
    auto c = instances_->instanceF(colorBuffer_, acc.id());
    c[0] = color.r;
    c[1] = color.g;
    c[2] = color.b;
    c[3] = color.a;
  });
}

} // poseidon namespace
