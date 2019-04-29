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

#ifndef POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_KERNELS_H
#define POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_KERNELS_H

#include <hermes/hermes.h>

namespace poseidon {

namespace cuda {
///
/// \param velocity
/// \param density
/// \param divergence
/// \param pressure
/// \param solid
/// \param forceField
/// \param solidVelocity
void bindTextures(const hermes::cuda::StaggeredGridTexture2 &velocity,
                  const hermes::cuda::GridTexture2<float> &density,
                  const hermes::cuda::GridTexture2<float> &divergence,
                  const hermes::cuda::GridTexture2<float> &pressure,
                  const hermes::cuda::GridTexture2<unsigned char> &solid,
                  const hermes::cuda::StaggeredGridTexture2 &forceField,
                  const hermes::cuda::StaggeredGridTexture2 &solidVelocity);
///
void unbindTextures();
///
void setupTextures();
///
/// \param velocity
/// \param forceField
/// \param dt
void applyForceField(hermes::cuda::StaggeredGridTexture2 &velocity,
                     const hermes::cuda::StaggeredGridTexture2 &forceField,
                     float dt);
///
/// \param velocity
/// \param solid
/// \param divergence
void computeDivergence(const hermes::cuda::StaggeredGridTexture2 &velocity,
                       const hermes::cuda::GridTexture2<unsigned char> &solid,
                       hermes::cuda::GridTexture2<float> &divergence);
///
/// \param divergence
/// \param solid
/// \param pressure
/// \param iterations
void computePressure(const hermes::cuda::GridTexture2<float> &divergence,
                     const hermes::cuda::GridTexture2<unsigned char> &solid,
                     hermes::cuda::GridTexture2<float> &pressure, float dt,
                     int iterations = 100);
///
/// \param pressure
/// \param solid
/// \param velocity
/// \param dt
void projectionStep(const hermes::cuda::GridTexture2<float> &pressure,
                    const hermes::cuda::GridTexture2<unsigned char> &solid,
                    hermes::cuda::StaggeredGridTexture2 &velocity, float dt);

} // namespace cuda

} // namespace poseidon

#endif