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
texture<float, cudaTextureType2D> phiNHatTex2;
texture<float, cudaTextureType2D> phiN1HatTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;

__global__ void __computePhiN1(float *phi, hermes::cuda::Grid2Info phiInfo,
                               hermes::cuda::Grid2Info uInfo,
                               hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * phiInfo.resolution.x + x;
  if (x < phiInfo.resolution.x && y < phiInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    unsigned char solid = tex2D(solidTex2, xc, yc);
    if (solid) {
      phi[index] = 0;
      return;
    }
    hermes::cuda::point2f wp = phiInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f up = uInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::point2f vp = vInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::point2f npos = phiInfo.toField(wp - vel * dt);
    hermes::cuda::point2f pos =
        hermes::cuda::floor(npos) + hermes::cuda::vec2(0.5);
    float nodeValues[4];
    nodeValues[0] = tex2D(phiTex2, pos.x, pos.y);
    nodeValues[1] = tex2D(phiTex2, pos.x + 1, pos.y);
    nodeValues[2] = tex2D(phiTex2, pos.x + 1, pos.y + 1);
    nodeValues[3] = tex2D(phiTex2, pos.x, pos.y + 1);
    float phiMin = min(nodeValues[3],
                       min(nodeValues[2], min(nodeValues[0], nodeValues[1])));
    float phiMax = max(nodeValues[3],
                       max(nodeValues[2], max(nodeValues[0], nodeValues[1])));
    phi[index] = tex2D(phiN1HatTex2, npos.x + 0.5, npos.y + 0.5) +
                 0.5 * (tex2D(phiTex2, xc, yc) - tex2D(phiNHatTex2, xc, yc));
    phi[index] = max(min(phi[index], phiMax), phiMin);
  }
}

MacCormackIntegrator2::MacCormackIntegrator2() {
  uTex2.filterMode = cudaFilterModeLinear;
  uTex2.normalized = 0;
  vTex2.filterMode = cudaFilterModeLinear;
  vTex2.normalized = 0;
  phiTex2.filterMode = cudaFilterModeLinear;
  phiTex2.normalized = 0;
  phiNHatTex2.filterMode = cudaFilterModeLinear;
  phiNHatTex2.normalized = 0;
  phiN1HatTex2.filterMode = cudaFilterModeLinear;
  phiN1HatTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
}

void MacCormackIntegrator2::set(hermes::cuda::Grid2Info info) {
  if (info.resolution != phiNHat.info().resolution)
    phiNHat.resize(info.resolution);
  phiNHat.setOrigin(info.origin);
  phiNHat.setDx(info.dx);
  if (info.resolution != phiN1Hat.info().resolution)
    phiN1Hat.resize(info.resolution);
  phiN1Hat.setOrigin(info.origin);
  phiN1Hat.setDx(info.dx);
  integrator.set(info);
}

void MacCormackIntegrator2::advect(
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
  CUDA_CHECK(cudaBindTextureToArray(
      phiNHatTex2, phiNHat.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      phiN1HatTex2, phiN1Hat.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex2, phi.texture().textureArray(),
                                    channelDesc));
  auto info = phi.info();
  hermes::ThreadArrayDistributionInfo td(info.resolution.x, info.resolution.y);
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, solid, phi, phiN1Hat, dt);
  CUDA_CHECK(cudaDeviceSynchronize());
  phiN1Hat.texture().updateTextureMemory();
  CUDA_CHECK(cudaDeviceSynchronize());
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, solid, phiN1Hat, phiNHat, -dt);
  CUDA_CHECK(cudaDeviceSynchronize());
  phiNHat.texture().updateTextureMemory();
  CUDA_CHECK(cudaDeviceSynchronize());
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  __computePhiN1<<<td.gridSize, td.blockSize>>>(phiOut.texture().deviceData(),
                                                phi.info(), velocity.u().info(),
                                                velocity.v().info(), dt);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
  cudaUnbindTexture(phiNHatTex2);
  cudaUnbindTexture(phiN1HatTex2);
}

} // namespace cuda

} // namespace poseidon