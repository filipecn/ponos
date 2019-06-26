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
texture<float, cudaTextureType2D> phiNHatTex2;
texture<float, cudaTextureType2D> phiN1HatTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;

__global__ void __computePhiN1_tt(float *phi, hermes::cuda::Grid2Info phiInfo,
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

__global__ void __computePhiN1_t(RegularGrid2Accessor<float> phi,
                                 RegularGrid2Info uInfo, RegularGrid2Info vInfo,
                                 float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (phi.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    unsigned char solid = tex2D(solidTex2, xc, yc);
    if (solid) {
      // phi(x, y) = 0;
      return;
    }
    point2f p = phi.worldPosition(x, y);
    point2f up = uInfo.toGrid(p) + vec2(0.5);
    point2f vp = vInfo.toGrid(p) + vec2(0.5);
    vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    point2f npos = phi.gridPosition(p - vel * dt);
    point2f pos = floor(npos) + vec2(0.5);
    float nodeValues[4];
    nodeValues[0] = tex2D(phiTex2, pos.x, pos.y);
    nodeValues[1] = tex2D(phiTex2, pos.x + 1, pos.y);
    nodeValues[2] = tex2D(phiTex2, pos.x + 1, pos.y + 1);
    nodeValues[3] = tex2D(phiTex2, pos.x, pos.y + 1);
    float phiMin = min(nodeValues[3],
                       min(nodeValues[2], min(nodeValues[0], nodeValues[1])));
    float phiMax = max(nodeValues[3],
                       max(nodeValues[2], max(nodeValues[0], nodeValues[1])));

    phi(x, y) = tex2D(phiN1HatTex2, xc, yc) +
                0.5 * (tex2D(phiTex2, xc, yc) - tex2D(phiNHatTex2, xc, yc));
    phi(x, y) = max(min(phi(x, y), phiMax), phiMin);
  }
}

__global__ void __computePhiN1(StaggeredGrid2Accessor vel,
                               RegularGrid2Accessor<unsigned char> solid,
                               RegularGrid2Accessor<float> solidPhi,
                               RegularGrid2Accessor<float> phiNHat,
                               RegularGrid2Accessor<float> phiN1Hat,
                               RegularGrid2Accessor<float> in,
                               RegularGrid2Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    if (solid(i, j)) {
      out(i, j) = solidPhi(i, j);
      return;
    }
    vec2f v = vel(i, j);
    point2f p = in.worldPosition(i, j);
    point2f wp = p - v * dt;
    point2f npos = in.gridPosition(wp);
    point2i pos(npos.x, npos.y);
    float nodeValues[4];
    nodeValues[0] = in(pos.x, pos.y);
    nodeValues[1] = in(pos.x + 1, pos.y);
    nodeValues[2] = in(pos.x + 1, pos.y + 1);
    nodeValues[3] = in(pos.x, pos.y + 1);
    float phiMin = min(nodeValues[3],
                       min(nodeValues[2], min(nodeValues[1], nodeValues[0])));
    float phiMax = max(nodeValues[3],
                       max(nodeValues[2], max(nodeValues[1], nodeValues[0])));

    out(i, j) = phiN1Hat(i, j) + 0.5 * (in(i, j) - phiNHat(i, j));
    out(i, j) = max(min(out(i, j), phiMax), phiMin);
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
  if (info.resolution != phiNHat_t.info().resolution)
    phiNHat_t.resize(info.resolution);
  phiNHat_t.setOrigin(info.origin);
  phiNHat_t.setDx(info.dx);
  if (info.resolution != phiN1Hat_t.info().resolution)
    phiN1Hat_t.resize(info.resolution);
  phiN1Hat_t.setOrigin(info.origin);
  phiN1Hat_t.setDx(info.dx);
  integrator.set(info);
}

void MacCormackIntegrator2::set(hermes::cuda::RegularGrid2Info info) {
  if (info.resolution != phiNHat.info().resolution)
    phiNHat.resize(info.resolution);
  phiNHat.setOrigin(info.origin);
  phiNHat.setSpacing(info.spacing);
  if (info.resolution != phiN1Hat.info().resolution)
    phiN1Hat.resize(info.resolution);
  phiN1Hat.setOrigin(info.origin);
  phiN1Hat.setSpacing(info.spacing);
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
      phiNHatTex2, phiNHat_t.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      phiN1HatTex2, phiN1Hat_t.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex2, phi.texture().textureArray(),
                                    channelDesc));
  auto info = phi.info();
  hermes::ThreadArrayDistributionInfo td(info.resolution.x, info.resolution.y);
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, solid, phi, phiN1Hat_t, dt);
  CUDA_CHECK(cudaDeviceSynchronize());
  phiN1Hat_t.texture().updateTextureMemory();
  CUDA_CHECK(cudaDeviceSynchronize());
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, solid, phiN1Hat_t, phiNHat_t, -dt);
  CUDA_CHECK(cudaDeviceSynchronize());
  phiNHat_t.texture().updateTextureMemory();
  CUDA_CHECK(cudaDeviceSynchronize());
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  __computePhiN1_tt<<<td.gridSize, td.blockSize>>>(
      phiOut.texture().deviceData(), phi.info(), velocity.u().info(),
      velocity.v().info(), dt);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
  cudaUnbindTexture(phiNHatTex2);
  cudaUnbindTexture(phiN1HatTex2);
}

void MacCormackIntegrator2::advect(StaggeredGrid2D &velocity,
                                   RegularGrid2Duc &solid,
                                   RegularGrid2Df &solidPhi,
                                   RegularGrid2Df &phi, RegularGrid2Df &phiOut,
                                   float dt) {
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, solid, solidPhi, phi, phiN1Hat, dt);
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, solid, solidPhi, phiN1Hat, phiNHat, -dt);
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solidPhi.accessor(),
      phiNHat.accessor(), phiN1Hat.accessor(), phi.accessor(),
      phiOut.accessor(), dt);
}

void MacCormackIntegrator2::advect_t(VectorGrid2D &velocity,
                                     RegularGrid2Duc &solid,
                                     RegularGrid2Df &phi,
                                     RegularGrid2Df &phiOut, float dt) {

  // phi^_n+1 = A(phi_n)
  integrator.advect_t(velocity, solid, phi, phiN1Hat, dt);
  Array2<float> phiN1HatArray(phiN1Hat.resolution());
  memcpy(phiN1HatArray, phiN1Hat.data());
  CUDA_CHECK(cudaDeviceSynchronize());

  // phi^_n = Ar(phi^_n+1)
  integrator.advect_t(velocity, solid, phiN1Hat, phiNHat, -dt);
  Array2<float> phiNHatArray(phiNHat.resolution());
  memcpy(phiNHatArray, phiNHat.data());
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(
      cudaBindTextureToArray(phiNHatTex2, phiNHatArray.data(), channelDesc));
  CUDA_CHECK(
      cudaBindTextureToArray(phiN1HatTex2, phiN1HatArray.data(), channelDesc));

  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1_t<<<td.gridSize, td.blockSize>>>(
      phiOut.accessor(), velocity.u().info(), velocity.v().info(), dt);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(phiTex2);
  cudaUnbindTexture(phiNHatTex2);
  cudaUnbindTexture(phiN1HatTex2);
}

__global__ void __computePhiN1(StaggeredGrid2Accessor vel,
                               RegularGrid2Accessor<float> phiNHat,
                               RegularGrid2Accessor<float> phiN1Hat,
                               RegularGrid2Accessor<float> in,
                               RegularGrid2Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    // if (solid(i, j)) {
    //   out(i, j) = solidPhi(i, j);
    //   return;
    // }
    vec2f v = vel(i, j);
    point2f p = in.worldPosition(i, j);
    point2f wp = p - v * dt;
    point2f npos = in.gridPosition(wp);
    point2i pos(npos.x, npos.y);
    float nodeValues[4];
    nodeValues[0] = in(pos.x, pos.y);
    nodeValues[1] = in(pos.x + 1, pos.y);
    nodeValues[2] = in(pos.x + 1, pos.y + 1);
    nodeValues[3] = in(pos.x, pos.y + 1);
    float phiMin = min(nodeValues[3],
                       min(nodeValues[2], min(nodeValues[1], nodeValues[0])));
    float phiMax = max(nodeValues[3],
                       max(nodeValues[2], max(nodeValues[1], nodeValues[0])));

    out(i, j) = phiN1Hat(i, j) + 0.5 * (in(i, j) - phiNHat(i, j));
    out(i, j) = max(min(out(i, j), phiMax), phiMin);
  }
}

void MacCormackIntegrator2::advect(StaggeredGrid2D &velocity,
                                   RegularGrid2Dm &material,
                                   RegularGrid2Df &phi, RegularGrid2Df &phiOut,
                                   float dt) {
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, material, phi, phiN1Hat, dt);
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, material, phiN1Hat, phiNHat, -dt);
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), phiNHat.accessor(), phiN1Hat.accessor(),
      phi.accessor(), phiOut.accessor(), dt);
}

} // namespace cuda

} // namespace poseidon