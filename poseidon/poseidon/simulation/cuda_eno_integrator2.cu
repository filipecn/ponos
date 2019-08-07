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

#include <hermes/numeric/cuda_interpolation.h>
#include <hermes/numeric/cuda_numeric.h>
#include <hermes/storage/cuda_storage_utils.h>
#include <poseidon/simulation/cuda_integrator.h>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

void ENOIntegrator2::set(hermes::cuda::RegularGrid2Info info) {}

/// Upwind + HJ ENO advection
__global__ void __eno_advect(StaggeredGrid2Accessor vel,
                             RegularGrid2Accessor<unsigned char> solid,
                             RegularGrid2Accessor<float> inSolid,
                             RegularGrid2Accessor<float> in,
                             RegularGrid2Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    if (solid(i, j)) {
      // out(i, j) = 0;
      return;
    }
    vec2f v = vel(i, j);
    vec2f dphi = gradientAt(in, i, j, 3, v);

    out(i, j) = in(i, j) - dt * dot(v, dphi);
    // out(i, j) = in(in.worldPosition(i, j) - dt * v);
  }
}

void ENOIntegrator2::advect(StaggeredGrid2D &velocity, RegularGrid2Duc &solid,
                            RegularGrid2Df &solidPhi, RegularGrid2Df &phi,
                            RegularGrid2Df &phiOut, float dt) {
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __eno_advect<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solidPhi.accessor(),
      phi.accessor(), phiOut.accessor(), dt);
}

/// Upwind + HJ ENO advection
__global__ void __eno_advect(StaggeredGrid2Accessor vel,
                             RegularGrid2Accessor<MaterialType> m,
                             RegularGrid2Accessor<float> in,
                             RegularGrid2Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    // if (m(i, j) == MaterialType::SOLID) {
    //   // out(i, j) = 0;
    //   return;
    // }
    //  else if (m(i, j) == MaterialType::FLUID) {
    vec2f v = vel(i, j);
    vec2f dphi = gradientAt(in, i, j, 2);
    printf("%f %f\n", dphi.x, dphi.y);
    out(i, j) = in(i, j) - dt * dot(v, dphi);
    // out(i, j) = in(in.worldPosition(i, j) - dt * v);
    // }
  }
}

void ENOIntegrator2::advect(hermes::cuda::StaggeredGrid2D &velocity,
                            RegularGrid2Dm &material,
                            hermes::cuda::RegularGrid2Df &in,
                            hermes::cuda::RegularGrid2Df &out, float dt) {
  hermes::ThreadArrayDistributionInfo td(in.resolution());
  __eno_advect<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), material.accessor(), in.accessor(), out.accessor(),
      dt);
}

} // namespace cuda

} // namespace poseidon