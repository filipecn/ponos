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

texture<float, cudaTextureType2D> uTex2;
texture<float, cudaTextureType2D> vTex2;
texture<float, cudaTextureType2D> phiTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;

__global__ void __advect(float *phi, hermes::cuda::Grid2Info phiInfo,
                         hermes::cuda::Grid2Info uInfo,
                         hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * phiInfo.resolution.x + x;
  if (x < phiInfo.resolution.x && y < phiInfo.resolution.y) {
    unsigned char solid = tex2D(solidTex2, x + 0.5, y + 0.5);
    if (solid) {
      phi[index] = 0;
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

void SemiLagrangianIntegrator2::advect(
    const hermes::cuda::VectorGridTexture2 &velocity,
    const hermes::cuda::GridTexture2<unsigned char> &solid,
    const hermes::cuda::GridTexture2<float> &phi,
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
  __advect<<<td.gridSize, td.blockSize>>>(phiOut.texture().deviceData(),
                                          phi.info(), velocity.u().info(),
                                          velocity.v().info(), dt);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
}

} // namespace cuda

} // namespace poseidon