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

//
// Created by FilipeCN on 2/18/2018.
//

#ifndef POSEIDON_PARTICLE_SYSTEM_MODEL_H
#define POSEIDON_PARTICLE_SYSTEM_MODEL_H

#include <aergia/aergia.h>
#include <poseidon/structures/particle_system.h>
#include <map>
#include <utility>

namespace poseidon {

/// Represents an scene object for particle systems
class ParticleSystemModel : public aergia::SceneObject {
public:
  /// \param ps particle system reference
  /// \param r **[opitional]** particle's radius (default = 0.03)
  explicit ParticleSystemModel(ParticleSystem &ps, float r = 0.03f);
  ~ParticleSystemModel() override;
  void draw() override;
  bool intersect(const ponos::Ray3 &r, float *t) override;
  /// Defines a color palette for a property
  /// \tparam T property type
  /// \param i property id
  /// \param p color palette
  template<typename T>
  void addPropertyColor(size_t i, aergia::ColorPalette p) {
    if (std::is_same<T, double>::value)
      paletteD_[i] = std::move(p);
  }
  template<typename T>
  void selectProperty(size_t i) {
    if (std::is_same<T, double>::value && paletteD_.find(i) != paletteD_.end())
      curProperty_ = static_cast<int>(i);
  }
  float particleRadius; ///< particle's sphere radius
  aergia::Color particleColor;
  /// update buffers
  void update();
private:
  int curProperty_; ///< current property to be drawn
  int activeParticle_; ///< selected particle
  std::map<size_t, aergia::ColorPalette> paletteD_; ///< color pallets for double type properties
  ParticleSystem &ps_; ///< particle system's reference
  // instances fields
  size_t posBuffer_, colorBuffer_;
  ponos::RawMeshSPtr particleMesh_;
  std::shared_ptr<aergia::SceneMesh> particleSceneMesh_;
  std::shared_ptr<aergia::InstanceSet> instances_;
};

} // poseidon namespace

#endif //POSEIDON_PARTICLE_SYSTEM_MODEL_H
