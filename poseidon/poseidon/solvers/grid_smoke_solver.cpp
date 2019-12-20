/// Copyright (c) 2019, FilipeCN.
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
///\file grid_smoke_solver.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-19
///
///\brief

#include <poseidon/solvers/grid_smoke_solver.h>
#include <poseidon/simulation/integrators/semi_lagrangian_integrator.h>

#include <memory>

using namespace ponos;

namespace poseidon {

GridSmokeSolver2::GridSmokeSolver2() {
  u_integrator_ = std::make_shared<SemiLagrangianIntegrator2>();
  v_integrator_ = std::make_shared<SemiLagrangianIntegrator2>();
  integrator_ = std::make_shared<SemiLagrangianIntegrator2>();
  addScalarField(); // 0 density
  addScalarField(); // 1 temperature
}

GridSmokeSolver2::~GridSmokeSolver2() = default;

void GridSmokeSolver2::setUIntegrator(Integrator2Interface *integrator) {
  u_integrator_.reset(integrator);
}

void GridSmokeSolver2::setVIntegrator(Integrator2Interface *integrator) {
  v_integrator_.reset(integrator);
}

void GridSmokeSolver2::setIntegrator(Integrator2Interface *integrator) {
  integrator_.reset(integrator);
}

void GridSmokeSolver2::setResolution(const size2 &res) {
  resolution_ = res;
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].resize(resolution_);
    for (auto &f : scalar_fields_[i])
      f.setResolution(resolution_);
  }
  for (auto &f : solid_scalar_fields_)
    f.setResolution(resolution_);
  vorticity_field_.resize(resolution_);
  pressure_.setResolution(resolution_);
  divergence_.resize(resolution_);
  material_.setResolution(resolution_);
  solid_velocity_.resize(resolution_);
  force_field_.resize(resolution_);
//  integrator_->set(scalarFields_[0][0].info());
//  u_integrator_->set(velocity_[0].u().info());
//  v_integrator_->set(velocity_[0].v().info());
//  pressureMatrix_.resize(resolution_);
//  scene_.resize(resolution_);
}

void GridSmokeSolver2::setSpacing(const vec2 &s) {
  spacing_ = s;
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].setSpacing(spacing_);
    for (auto &f : scalar_fields_[i])
      f.setSpacing(spacing_);
  }
  for (auto &f : solid_scalar_fields_)
    f.setSpacing(spacing_);
  vorticity_field_.setSpacing(spacing_);
  pressure_.setSpacing(spacing_);
  solid_velocity_.setSpacing(spacing_);
  force_field_.setSpacing(spacing_);
//  integrator_->set(scalarFields_[0][0].info());
//  uIntegrator_->set(velocity_[0].u().info());
//  vIntegrator_->set(velocity_[0].v().info());
//  scene_.setSpacing(spacing_);
}

void GridSmokeSolver2::setOrigin(const point2 &o) {
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].setOrigin(o);
    for (auto &f : scalar_fields_[i])
      f.setOrigin(o);
  }
  for (auto &f : solid_scalar_fields_)
    f.setOrigin(o);
  vorticity_field_.setOrigin(o);
  pressure_.setOrigin(o);
//  divergence_.setOrigin(o);
  material_.setOrigin(o);
  solid_velocity_.setOrigin(o);
  force_field_.setOrigin(o);
//  integrator_->set(scalarFields_[0][0].info());
//  uIntegrator_->set(velocity_[0].u().info());
//  vIntegrator_->set(velocity_[0].v().info());
}

size_t GridSmokeSolver2::addScalarField() {
  scalar_fields_[src].emplace_back();
  scalar_fields_[dst].emplace_back();
  solid_scalar_fields_.emplace_back();
  solid_scalar_fields_.emplace_back();
  return scalar_fields_[src].size() - 1;
}

void GridSmokeSolver2::step(float dt) {
  // rasterColliders();
  // uIntegrator_->advect(velocity_[src], solid_, solidVelocity_.u(),
  //                      velocity_[src].u(), velocity_[dst].u(), dt);
  // vIntegrator_->advect(velocity_[src], solid_, solidVelocity_.v(),
  //                      velocity_[src].v(), velocity_[dst].v(), dt);
  velocity_[dst].copy(velocity_[src]);
  for (size_t i = 0; i < scalar_fields_[src].size(); i++)
    integrator_->advect(velocity_[dst].accessor(),
                        material_.accessor(),
                        scalar_fields_[src][i].accessor(),
                        scalar_fields_[dst][i].accessor(),
                        dt);
  src = src ? 0 : 1;
  dst = dst ? 0 : 1;
  force_field_ = 0.f;
//  addBuoyancyForce(forceField_, solid_, scalarFields_[src][0],
//                   scalarFields_[src][1], 273, 1.0f, 0.0f);
  // addVorticityConfinementForce(forceField_, velocity_[src],
  // solid_,vorticityField_, 2.f, dt);
  // applyForceField(velocity_[src], solid_, forceField_, dt);
  // injectSmoke(scalarFields_[src][0], scene_.smoke_source, dt);
  // injectTemperature(scalarFields_[src][1], scene_.target_temperature, dt);
//  computeDivergence(velocity_[src], solid_, solidVelocity_, divergence_);
//  solvePressureSystem(pressureMatrix_, divergence_, pressure_, solid_, dt);
//  projectionStep_t(pressure_, solid_, velocity_[src], dt);
}

Grid2Accessor<real_t> GridSmokeSolver2::scalarField(size_t i) {
  return scalar_fields_[src][i].accessor();
}

StaggeredGrid2Accessor GridSmokeSolver2::velocity() { return velocity_[src].accessor(); }

Grid2Accessor<MaterialType> GridSmokeSolver2::material() { return material_.accessor(); }

MemoryBlock2<real_t> &GridSmokeSolver2::divergence() { return divergence_; }

} // poseidon namespace