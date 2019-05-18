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

#ifndef POSEIDON_SOLVERS_CUDA_SMOKE3_SOLVER_KERNELS_H
#define POSEIDON_SOLVERS_CUDA_SMOKE3_SOLVER_KERNELS_H

#include <hermes/hermes.h>
#include <poseidon/math/cuda_fd.h>

namespace poseidon {

namespace cuda {

template <hermes::cuda::MemoryLocation L>
void injectTemperature(hermes::cuda::RegularGrid3<L, float> &temperature,
                       hermes::cuda::RegularGrid3<L, float> &targetTemperature,
                       float dt);
template <hermes::cuda::MemoryLocation L>
void injectSmoke(hermes::cuda::RegularGrid3<L, float> &smoke,
                 hermes::cuda::RegularGrid3<L, unsigned char> &source,
                 float dt);
/// \param velocity
/// \param forceField
/// \param dt
template <hermes::cuda::MemoryLocation L>
void applyForceField(hermes::cuda::StaggeredGrid3<L> &velocity,
                     hermes::cuda::VectorGrid3<L> &forceField, float dt);
template <hermes::cuda::MemoryLocation L>
void applyBuoyancyForceField(hermes::cuda::StaggeredGrid3<L> &velocity,
                             hermes::cuda::RegularGrid3<L, float> &density,
                             hermes::cuda::RegularGrid3<L, float> &temperature,
                             float ambientTemperature, float alpha, float beta,
                             float dt);
///
/// \param velocity
/// \param solid
/// \param divergence
template <hermes::cuda::MemoryLocation L>
void computeDivergence(hermes::cuda::StaggeredGrid3<L> &velocity,
                       hermes::cuda::RegularGrid3<L, unsigned char> &solid,
                       hermes::cuda::RegularGrid3<L, float> &divergence);
template <hermes::cuda::MemoryLocation L>
size_t setupPressureSystem(hermes::cuda::RegularGrid3<L, float> &divergence,
                           hermes::cuda::RegularGrid3<L, unsigned char> &solid,
                           FDMatrix3<L> &A, float dt,
                           hermes::cuda::MemoryBlock1<L, float> &rhs);
///
template <hermes::cuda::MemoryLocation L>
void solvePressureSystem(FDMatrix3<L> &pressureMatrix,
                         hermes::cuda::RegularGrid3<L, float> &divergence,
                         hermes::cuda::RegularGrid3<L, float> &pressure,
                         hermes::cuda::RegularGrid3<L, unsigned char> &solid,
                         float dt);
///
/// \param pressure
/// \param solid
/// \param velocity
/// \param dt
template <hermes::cuda::MemoryLocation L>
void projectionStep(hermes::cuda::RegularGrid3<L, float> &pressure,
                    hermes::cuda::RegularGrid3<L, unsigned char> &solid,
                    hermes::cuda::StaggeredGrid3<L> &velocity, float dt);

} // namespace cuda

} // namespace poseidon

#endif