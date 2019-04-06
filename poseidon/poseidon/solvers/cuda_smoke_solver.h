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

#ifndef POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H
#define POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H

#include <hermes/hermes.h>
#include <ponos/geometry/point.h>

namespace poseidon {

namespace cuda {

/// Eulerian grid based solver for smoke simulations. Stores its data in fast
/// device texture memory.
class GridSmokeSolver2 {
public:
  GridSmokeSolver2() = default;
  ~GridSmokeSolver2();
  void init();
  ///
  ///
  /// \param res
  void setResolution(const ponos::uivec2 &res);
  /// Sets cell size
  /// \param dx scale
  void setDx(float dx);
  /// Sets lower left corner position
  /// \param o offset
  void setOrigin(const ponos::point2f &o);
  /// Advances one simulation step
  /// \param dt time step
  void step(float dt);

  hermes::cuda::StaggeredGridTexture2 &velocityData();
  /// \return const hermes::cuda::GridTexture2<float>& density data reference
  const hermes::cuda::GridTexture2<float> &densityData() const;
  /// \return hermes::cuda::GridTexture2<float>& density data reference
  hermes::cuda::GridTexture2<float> &densityData();

protected:
  void setupTextures();
  void advectVelocities(float dt);
  void advectDensity(float dt);

  hermes::cuda::StaggeredGridTexture2 velocity;
  hermes::cuda::GridTexture2<float> density;
  hermes::cuda::GridTexture2<float> pressure;
  hermes::cuda::GridTexture2<float> divergence;
  hermes::cuda::vec2u resolution;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H