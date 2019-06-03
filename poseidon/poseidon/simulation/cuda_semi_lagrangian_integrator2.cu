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

#include <poseidon/simulation/cuda_integrator.h>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

texture<float, cudaTextureType2D> uTex2;
texture<float, cudaTextureType2D> vTex2;
texture<float, cudaTextureType2D> phiTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;

__global__ void __advect_tt(float *phi, hermes::cuda::Grid2Info phiInfo,
                            hermes::cuda::Grid2Info uInfo,
                            hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * phiInfo.resolution.x + x;
  if (x < phiInfo.resolution.x && y < phiInfo.resolution.y) {
    unsigned char solid = tex2D(solidTex2, x + 0.5, y + 0.5);
    if (solid) {
      // phi[index] = 0;
      return;
    }
    hermes::cuda::point2f wp = phiInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f up = uInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::point2f vp = vInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::point2f pos =
        phiInfo.toField(wp - vel * dt) + hermes::cuda::vec2(0.5);
    phi[index] = tex2D(phiTex2, pos.x, pos.y);
  }
}

__global__ void __advect_t(RegularGrid2Accessor<float> phi,
                           RegularGrid2Accessor<unsigned char> solid,
                           RegularGrid2Info uInfo, RegularGrid2Info vInfo,
                           float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (phi.isIndexStored(x, y)) {
    if (solid(x, y)) {
      // phi(x, y, z) = 0;
      return;
    }
    point2f p = phi.worldPosition(x, y);
    point2f up = uInfo.toGrid(p) + vec2(0.5);
    point2f vp = vInfo.toGrid(p) + vec2(0.5);
    vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    point2f pos = phi.gridPosition(p - vel * dt) + vec2(0.5);
    phi(x, y) = tex2D(phiTex2, pos.x, pos.y);
  }
}

__global__ void __advect(StaggeredGrid2Accessor vel,
                         RegularGrid2Accessor<unsigned char> solid,
                         RegularGrid2Accessor<float> solidPhi,
                         RegularGrid2Accessor<float> in,
                         RegularGrid2Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    if (solid(i, j)) {
      out(i, j) = solidPhi(i, j);
      return;
    }
    point2f p = in.worldPosition(i, j);
    vec2f v = vel(i, j);
    point2f pos = p - v * dt;
    // TODO: clip on solid walls
    out(i, j) = in(pos);
  }
}

SemiLagrangianIntegrator2::SemiLagrangianIntegrator2() {
  uTex2.filterMode = cudaFilterModeLinear;
  uTex2.normalized = 0;
  vTex2.filterMode = cudaFilterModeLinear;
  vTex2.normalized = 0;
  phiTex2.filterMode = cudaFilterModeLinear;
  phiTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
}

// TODO: DEPRECATED
void SemiLagrangianIntegrator2::advect(
    hermes::cuda::VectorGridTexture2 &velocity,
    hermes::cuda::GridTexture2<unsigned char> &solid,
    hermes::cuda::GridTexture2<float> &phi,
    hermes::cuda::GridTexture2<float> &phiOut, float dt) {
  using namespace hermes::cuda;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, solid.texture().textureArray(),
                                    channelDesc));
  channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(
      uTex2, velocity.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vTex2, velocity.v().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex2, phi.texture().textureArray(),
                                    channelDesc));
  auto info = phi.info();
  hermes::ThreadArrayDistributionInfo td(info.resolution.x, info.resolution.y);
  __advect_tt<<<td.gridSize, td.blockSize>>>(phiOut.texture().deviceData(),
                                             phi.info(), velocity.u().info(),
                                             velocity.v().info(), dt);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
}

void SemiLagrangianIntegrator2::advect(StaggeredGrid2D &velocity,
                                       RegularGrid2Duc &solid,
                                       RegularGrid2Df &solidPhi,
                                       RegularGrid2Df &phi,
                                       RegularGrid2Df &phiOut, float dt) {
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __advect<<<td.gridSize, td.blockSize>>>(velocity.accessor(), solid.accessor(),
                                          solidPhi.accessor(), phi.accessor(),
                                          phiOut.accessor(), dt);
}

void SemiLagrangianIntegrator2::advect_t(VectorGrid2D &velocity,
                                         RegularGrid2Duc &solid,
                                         RegularGrid2Df &phi,
                                         RegularGrid2Df &phiOut, float dt) {
  Array2<unsigned char> solidArray(solid.resolution());
  Array2<float> uArray(velocity.u().resolution());
  Array2<float> vArray(velocity.v().resolution());
  Array2<float> phiArray(phi.resolution());
  memcpy(solidArray, solid.data());
  memcpy(uArray, velocity.u().data());
  memcpy(vArray, velocity.v().data());
  memcpy(phiArray, phi.data());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, solidArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(uTex2, uArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(vTex2, vArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex2, phiArray.data(), channelDesc));
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __advect_t<<<td.gridSize, td.blockSize>>>(phiOut.accessor(), solid.accessor(),
                                            velocity.u().info(),
                                            velocity.v().info(), dt);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
}
} // namespace cuda

} // namespace poseidon