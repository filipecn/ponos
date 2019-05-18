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

#include <hermes/storage/cuda_storage_utils.h>
#include <poseidon/simulation/cuda_integrator.h>

namespace poseidon {

namespace cuda {

texture<float, cudaTextureType3D> uTex3;
texture<float, cudaTextureType3D> vTex3;
texture<float, cudaTextureType3D> wTex3;
texture<float, cudaTextureType3D> phiTex3;
texture<unsigned char, cudaTextureType3D> solidTex3;

__global__ void __advect(hermes::cuda::RegularGrid3Accessor<float> phi,
                         hermes::cuda::RegularGrid3Info uInfo,
                         hermes::cuda::RegularGrid3Info vInfo,
                         hermes::cuda::RegularGrid3Info wInfo, float dt) {
  using namespace hermes::cuda;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (phi.isIndexStored(x, y, z)) {
    unsigned char solid = tex3D(solidTex3, x + 0.5, y + 0.5, z + 0.5);
    if (solid) {
      phi(x, y, z) = 0;
      return;
    }
    point3f p = phi.worldPosition(x, y, z);
    point3f up = uInfo.toGrid(p) + vec3(0.5);
    point3f vp = vInfo.toGrid(p) + vec3(0.5);
    point3f wp = wInfo.toGrid(p) + vec3(0.5);
    vec3f vel(tex3D(uTex3, up.x, up.y, up.z), tex3D(vTex3, vp.x, vp.y, vp.z),
              tex3D(wTex3, vp.x, wp.y, wp.z));
    point3f pos = phi.gridPosition(p - vel * dt) + vec3(0.5);
    phi(x, y, z) = tex3D(phiTex3, pos.x, pos.y, pos.z);
  }
}

SemiLagrangianIntegrator3::SemiLagrangianIntegrator3() {
  uTex3.filterMode = cudaFilterModeLinear;
  uTex3.normalized = 0;
  vTex3.filterMode = cudaFilterModeLinear;
  vTex3.normalized = 0;
  wTex3.filterMode = cudaFilterModeLinear;
  wTex3.normalized = 0;
  phiTex3.filterMode = cudaFilterModeLinear;
  phiTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
}

void SemiLagrangianIntegrator3::advect(
    hermes::cuda::VectorGrid3D &velocity,
    hermes::cuda::RegularGrid3<hermes::cuda::MemoryLocation::DEVICE,
                               unsigned char> &solid,
    hermes::cuda::RegularGrid3Df &phi, hermes::cuda::RegularGrid3Df &phiOut,
    float dt) {
  using namespace hermes::cuda;
  Array3<unsigned char> solidArray(solid.resolution());
  Array3<float> uArray(velocity.u().resolution());
  Array3<float> vArray(velocity.v().resolution());
  Array3<float> wArray(velocity.w().resolution());
  Array3<float> phiArray(phi.resolution());
  memcpy(solidArray, solid.data());
  memcpy(uArray, velocity.u().data());
  memcpy(vArray, velocity.v().data());
  memcpy(wArray, velocity.w().data());
  memcpy(phiArray, phi.data());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, solidArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(uTex3, uArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(vTex3, vArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wTex3, wArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex3, phiArray.data(), channelDesc));
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __advect<<<td.gridSize, td.blockSize>>>(
      phiOut.accessor(), velocity.u().info(), velocity.v().info(),
      velocity.w().info(), dt);
  cudaUnbindTexture(solidTex3);
  cudaUnbindTexture(vTex3);
  cudaUnbindTexture(uTex3);
  cudaUnbindTexture(wTex3);
  cudaUnbindTexture(phiTex3);
}

} // namespace cuda

} // namespace poseidon