/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
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

#ifndef POSEIDON_SOLVERS_CUDA_PRACTICAL_LIQUIDS_SOLVER3_H
#define POSEIDON_SOLVERS_CUDA_PRACTICAL_LIQUIDS_SOLVER3_H

#include <hermes/numeric/cuda_grid.h>
#include <hermes/parallel/cuda_reduce.h>
#include <ponos/common/timer.h>
#include <poseidon/math/cuda_fd.h>
#include <poseidon/simulation/cuda_defs.h>
#include <poseidon/simulation/cuda_integrator.h>
#include <poseidon/simulation/cuda_level_set.h>
#include <poseidon/simulation/cuda_particle_system2.h>

namespace poseidon {

namespace cuda {
// Discretization goes as follows:
// - velocity components are stored on face centers (staggered)
// - level set distances are stored on cell centers
// - all other fields are stored on cell centers
// So, for a NxMxO resolution simulation, the sizes of the grids are:
// - N+1xMxO for u velocity component
// - NxM+1xO for v velocity component
// - NxMxO+1 for w velocity component
// - NxMxO for level set distances
// - NxMxO for other fields
// For cell i,j,k:
//              v(i,j+1,k)
//           --------------
//          |              |
//          |              |
// u(i,j,k) |   p(i,j,k)   | u(i+1,j,k)
//          |              |
//          |              |
//           --------------
//              v(i,j,k)
template <hermes::cuda::MemoryLocation L> class PracticalLiquidsSolver3 {
public:
  PracticalLiquidsSolver3() {
    setSpacing(hermes::cuda::vec3f(1.f));
    setOrigin(hermes::cuda::point3f(0.f));
  }
  /// \param res **[in]** resolution in number of cells
  void setResolution(const hermes::cuda::vec3u &res) {
    for (int i = 0; i < 3; i++) {
      velocity_[i].resize(res);
      surface_ls_[i].setResolution(res);
    }
    material_.resize(res);
    solid_velocity_.resize(res);
    pressure_.resize(res);
    divergence_.resize(res);
    solid_ls_.setResolution(res);
    force_field_.resize(res);
    pressure_matrix_.resize(res);
    init();
  }
  /// \param s **[in]** distance between two cell centers
  void setSpacing(const hermes::cuda::vec3f &s) {
    for (int i = 0; i < 3; i++) {
      velocity_[i].setSpacing(s);
      surface_ls_[i].setSpacing(s);
    }
    solid_velocity_.setSpacing(s);
    material_.setSpacing(s);
    pressure_.setSpacing(s);
    divergence_.setSpacing(s);
    solid_ls_.setSpacing(s);
    force_field_.setSpacing(s);
  }
  /// \param o **[in]** world position of lowest cell center
  void setOrigin(const hermes::cuda::point3f &o) {
    for (int i = 0; i < 3; i++) {
      velocity_[i].setOrigin(o);
      surface_ls_[i].setOrigin(o);
    }
    solid_velocity_.setOrigin(o);
    material_.setOrigin(o);
    pressure_.setOrigin(o);
    divergence_.setOrigin(o);
    solid_ls_.setOrigin(o);
    force_field_.setOrigin(o);
  }
  void init() {
    hermes::cuda::fill3(velocity_[SRC].u().data(), 0.f);
    hermes::cuda::fill3(velocity_[SRC].v().data(), 0.f);
    hermes::cuda::fill3(velocity_[SRC].w().data(), 0.f);
    hermes::cuda::fill3(force_field_.u().data(), 0.f);
    hermes::cuda::fill3(force_field_.v().data(), 0.f);
    hermes::cuda::fill3(force_field_.w().data(), 0.f);
    hermes::cuda::fill3(solid_ls_.grid().data(),
                        hermes::cuda::Constants::greatest<float>());
    fluidIntegrator.set(pressure_.info());
  }
  /// \param box **[in]**
  /// \param velocity **[in]**
  void rasterSolid(const hermes::cuda::bbox3f &box,
                   const hermes::cuda::vec3f &velocity);
  /// \param time_step **[in]**
  void step(float time_step) {
    while (time_step > 0) {
      float dt = min(time_step, dtFromCFLCondition());
      std::cerr << "timestep: " << dt << std::endl;
      ponos::Timer timer;
      // input: LS[SRC] output: LS[DST]
      updateSurfaceDistanceField();
      std::cerr << "updateSurfaceDistanceFiled in " << timer.tackTick()
                << "ms\n";
      // input: LS[DST] output: material
      classifyCells();
      std::cerr << "classifyCells in " << timer.tackTick() << "ms\n";
      fill3(velocity_[DST].u().data(), 0.f);
      fill3(velocity_[DST].v().data(), 0.f);
      fill3(velocity_[DST].w().data(), 0.f);
      fill3(divergence_.data(), 0.f);
      fill3(pressure_.data(), 0.f);
      std::cerr << "resetFields in " << timer.tackTick() << "ms\n";
      // input: material, force, v[SRC] output: v[DST]
      // std::cerr << "SROUCE V\n";
      // std::cerr << velocity_[SRC].v().data() << std::endl;
      timer.tackTick();
      applyExternalForces(dt);
      std::cerr << "applyExternalForces in " << timer.tackTick() << "ms\n";
      // input: material, v[DST] output: v[SRC]
      propagateVelocity();
      std::cerr << "propagateVelocity in " << timer.tackTick() << "ms\n";
      // velocity_[DST].copy(velocity_[SRC]);
      advectVelocity(dt);
      std::cerr << "advectVelocity in " << timer.tackTick() << "ms\n";
      // velocity_[SRC].copy(velocity_[DST]);
      // inpute materia, v[DST] output: v[DST]
      enforceBoundaries(DST, false);
      std::cerr << "enforceBoundaries in " << timer.tackTick() << "ms\n";
      // resolveViscosity(dt);
      // input: v[DST] output: divergence
      computeDivergence();
      std::cerr << "computeDivergence in " << timer.tackTick() << "ms\n";
      // std::cerr << divergence_.data() << std::endl;
      // input: material, divergence output: pressure
      solvePressure(dt);
      std::cerr << "solvePressure in " << timer.tackTick() << "ms\n";
      // std::cerr << material_.data() << std::endl;
      // std::cerr << divergence_.data() << std::endl;
      // std::cerr << pressure_.data() << std::endl;
      // input: v[DST] output: v[DST]
      project(dt);
      std::cerr << "project in " << timer.tackTick() << "ms\n";
      // velocity_[DST].copy(velocity_[SRC]);
      // input: material, v[DST] output: v[SRC]
      propagateVelocity();
      std::cerr << "propagateVelocity in " << timer.tackTick() << "ms\n";
      // input: material, v[SRC] output: v[SRC]
      enforceBoundaries(SRC, true);
      std::cerr << "enforceBoundaries in " << timer.tackTick() << "ms\n";
      // velocity_[SRC].copy(velocity_[DST]);
      // input: material, v[SRC], LS[SRC] output: LS[DST]
      advectFluid(dt);
      // dtFromCFLCondition();
      std::cerr << "advectFluid in " << timer.tackTick() << "ms\n";
      velocity_[DST].copy(velocity_[SRC]);
      std::cerr << "velocity copy in " << timer.tackTick() << "ms\n";
      time_step -= dt;
      SRC = (SRC) ? 0 : 1;
      DST = (DST) ? 0 : 1;
    }
  }
  LevelSet3<L> &surfaceLevelSet() { return surface_ls_[SRC]; }
  LevelSet3<L> &solidLevelSet() { return solid_ls_; }
  hermes::cuda::VectorGrid3<L> &forceField() { return force_field_; }
  hermes::cuda::StaggeredGrid3<L> &velocity() { return velocity_[SRC]; }

private:
  /// \return float maximum value allowed for dt
  float dtFromCFLCondition() {
    using namespace hermes::cuda;
    float min_s = fminf(material_.spacing().x,
                        fminf(material_.spacing().y, material_.spacing().z));
    float max_v = fmaxf(maxAbs(velocity_[SRC].u().data()),
                        fmaxf(maxAbs(velocity_[SRC].w().data()),
                              maxAbs(velocity_[SRC].v().data())));
    std::cerr << "max_v " << max_v << std::endl;
    if (max_v < 1e-8)
      return Constants::greatest<float>();
    return min_s / max_v;
  }
  /// Propagate distances from liquid surface
  void updateSurfaceDistanceField();
  /// Given a level set distance defined over the entire domain,
  /// classify cells in FLUID, AIR or SOLID
  void classifyCells();
  /// Apply an external force field to fluid cells
  /// \param dt **[in]**
  void applyExternalForces(float dt);
  /// \param dt **[in]**
  void computeDivergence();
  /// \param dt **[in]**
  void solvePressure(float dt);
  /// \param dt **[in]**
  void project(float dt);
  /// \param dt **[in]**
  void advectVelocity(float dt);
  /// \param dt **[in]**
  void advectFluid(float dt);
  /// Propagates velocities throughout a narrow band outside fluid.
  /// The surface distance level set must be previously computed
  void propagateVelocity();
  // After velocity advection, solid walls may end up with wrong velocities
  // This function sets velocity components values on solid walls appropriately
  void enforceBoundaries(size_t buffer, bool onlyNormalVelocities);

  size_t SRC = 0;
  size_t DST = 1;
  hermes::cuda::StaggeredGrid3<L> velocity_[2];
  hermes::cuda::StaggeredGrid3<L> solid_velocity_;
  hermes::cuda::VectorGrid3<L> force_field_;
  LevelSet3<L> surface_ls_[2];
  LevelSet3<L> solid_ls_;
  FDMatrix3<L> pressure_matrix_;
  hermes::cuda::RegularGrid3<L, float> pressure_;
  hermes::cuda::RegularGrid3<L, float> divergence_;
  hermes::cuda::RegularGrid3<L, MaterialType> material_;
  SemiLagrangianIntegrator3 fluidIntegrator;
};

using PracticalLiquidsSolver3D =
    PracticalLiquidsSolver3<hermes::cuda::MemoryLocation::DEVICE>;
using PracticalLiquidsSolver3H =
    PracticalLiquidsSolver3<hermes::cuda::MemoryLocation::HOST>;

} // namespace cuda

} // namespace poseidon

#endif