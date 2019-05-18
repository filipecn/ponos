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

#include <poseidon/simulation/cuda_vector_fields.h>

namespace poseidon {

namespace cuda {

__host__ __device__ hermes::cuda::vec3f
enrightDeformationField(hermes::cuda::point3f p) {
  float pix = hermes::cuda::Constants::pi() * p.x;
  float piy = hermes::cuda::Constants::pi() * p.y;
  float piz = hermes::cuda::Constants::pi() * p.z;
  return hermes::cuda::vec3f(
      2. * sinf(pix) * sinf(pix) * sinf(2 * piy) * sinf(2 * piz),
      -sinf(2 * pix) * sinf(piy) * sinf(piy) * sinf(2 * piz),
      -sinf(2 * pix) * sinf(2 * piy) * sinf(piz) * sinf(piz));
}

__global__ void
__enrightComponent(hermes::cuda::RegularGrid3Accessor<float> acc,
                   size_t component) {
  using namespace hermes::cuda;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < acc.resolution().x && y < acc.resolution().y &&
      z < acc.resolution().z) {
    acc(x, y, z) =
        enrightDeformationField(acc.worldPosition(x, y, z))[component];
  }
}

void applyEnrightDeformationField(hermes::cuda::StaggeredGrid3D &grid) {
  {
    hermes::ThreadArrayDistributionInfo td(grid.u().resolution());
    __enrightComponent<<<td.gridSize, td.blockSize>>>(grid.u().accessor(), 0);
  }
  {
    hermes::ThreadArrayDistributionInfo td(grid.v().resolution());
    __enrightComponent<<<td.gridSize, td.blockSize>>>(grid.v().accessor(), 1);
  }
  {
    hermes::ThreadArrayDistributionInfo td(grid.w().resolution());
    __enrightComponent<<<td.gridSize, td.blockSize>>>(grid.w().accessor(), 2);
  }
}

} // namespace cuda

} // namespace poseidon