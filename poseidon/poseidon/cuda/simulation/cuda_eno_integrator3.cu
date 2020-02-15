/*
 * Copyright (c) 3019 FilipeCN
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

void ENOIntegrator3::set(hermes::cuda::RegularGrid3Info info) {}

/// Upwind + HJ ENO advection
__global__ void __eno_advect(StaggeredGrid3Accessor vel,
                             RegularGrid3Accessor<unsigned char> solid,
                             RegularGrid3Accessor<float> inSolid,
                             RegularGrid3Accessor<float> in,
                             RegularGrid3Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (in.isIndexStored(i, j, k)) {
    if (solid(i, j, k)) {
      // out(i, j) = 0;
      return;
    }
    vec3f v = vel(i, j, k);
    vec3f dphi = gradientAt(in, i, j, k, 3, v);

    out(i, j, k) = in(i, j, k) - dt * dot(v, dphi);
    // out(i, j) = in(in.worldPosition(i, j) - dt * v);
  }
}

void ENOIntegrator3::advect(StaggeredGrid3D &velocity, RegularGrid3Duc &solid,
                            RegularGrid3Df &solidPhi, RegularGrid3Df &phi,
                            RegularGrid3Df &phiOut, float dt) {
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __eno_advect<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solidPhi.accessor(),
      phi.accessor(), phiOut.accessor(), dt);
}

/// Upwind + HJ ENO advection
__global__ void __eno_advect(StaggeredGrid3Accessor vel,
                             RegularGrid3Accessor<MaterialType> m,
                             RegularGrid3Accessor<float> in,
                             RegularGrid3Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (in.isIndexStored(i, j, k)) {
    if (m(i, j, k) == MaterialType::SOLID) {
      // out(i, j) = 0;
      return;
    }
    //  else if (m(i, j) == MaterialType::FLUID) {
    vec3f v = vel(i, j, k);
    vec3f dphi = gradientAt(in, i, j, k, 1, v);

    out(i, j, k) = in(i, j, k) - dt * dot(v, dphi);
    out(i, j, k) = in(in.worldPosition(i, j, k) - dt * v);
    // }
  }
}

void ENOIntegrator3::advect(hermes::cuda::StaggeredGrid3D &velocity,
                            RegularGrid3Dm &material,
                            hermes::cuda::RegularGrid3Df &in,
                            hermes::cuda::RegularGrid3Df &out, float dt) {
  hermes::ThreadArrayDistributionInfo td(in.resolution());
  __eno_advect<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), material.accessor(), in.accessor(), out.accessor(),
      dt);
}

} // namespace cuda

} // namespace poseidon