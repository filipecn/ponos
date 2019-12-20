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
///\file smoke_solver.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-19
///
///\brief

#ifndef POSEIDON_SOLVERS_SMOKE_SOLVER_H
#define POSEIDON_SOLVERS_SMOKE_SOLVER_H

#include <poseidon/numeric/staggered_grid.h>
#include <poseidon/simulation/definitions.h>
#include <poseidon/simulation/integrators/integrator_interface.h>

namespace poseidon {

/// Eulerian grid based solver for smoke simulations.
class GridSmokeSolver2 {
public:
  GridSmokeSolver2();
  virtual ~GridSmokeSolver2();
  void setUIntegrator(Integrator2Interface *integrator);
  void setVIntegrator(Integrator2Interface *integrator);
  void setIntegrator(Integrator2Interface *integrator);
  void setResolution(const ponos::size2 &resolution);
  void setSpacing(const ponos::vec2 &spacing);
  void setOrigin(const ponos::point2 &origin);
  size_t addScalarField();
  void step(real_t dt);
  StaggeredGrid2Accessor velocity();
  ponos::MemoryBlock2<real_t> &divergence();
  ponos::Grid2Accessor<MaterialType> material();
  ponos::Grid2Accessor<real_t> scalarField(size_t i);
private:
//  Scene3<float> scene_;
  std::shared_ptr<Integrator2Interface> u_integrator_;
  std::shared_ptr<Integrator2Interface> v_integrator_;
  std::shared_ptr<Integrator2Interface> integrator_;
//  poseidon::cuda::FDMatrix3D pressureMatrix_;
  StaggeredGrid2 velocity_[2];
  StaggeredGrid2 solid_velocity_;
  VectorGrid2 force_field_, vorticity_field_;
  ponos::Grid2<real_t> pressure_;
  ponos::MemoryBlock2<real_t> divergence_;
  ponos::Grid2<MaterialType> material_;
  std::vector<ponos::Grid2<real_t>> scalar_fields_[2];
  std::vector<ponos::Grid2<real_t>> solid_scalar_fields_;
  ponos::size2 resolution_;
  ponos::vec2 spacing_;
  size_t src = 0;
  size_t dst = 1;
};

} // poseidon namespace

#endif //POSEIDON_SOLVERS_SMOKE_SOLVER_H