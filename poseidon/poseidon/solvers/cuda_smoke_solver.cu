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

#include <hermes/common/cuda.h>
#include <poseidon/solvers/cuda_smoke_solver.h>

namespace poseidon {

namespace cuda {

texture<float, cudaTextureType2D> uTex2;
texture<float, cudaTextureType2D> vTex2;
texture<float, cudaTextureType2D> densityTex2;
texture<float, cudaTextureType2D> pressureTex2;
texture<float, cudaTextureType2D> divergenceTex2;

__global__ void __advectUVelocities(float *u, hermes::cuda::Grid2Info uInfo,
                                    hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * uInfo.resolution.x + x;
  if (x < uInfo.resolution.x && y < uInfo.resolution.y) {
    hermes::cuda::vec2f wp = uInfo.toWorld(hermes::cuda::vec2f(x, y));
    hermes::cuda::vec2f vp = vInfo.toField(wp);
    hermes::cuda::vec2f vel(tex2D(uTex2, x, y), tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::vec2f pos = wp - vel * dt;
    u[index] = tex2D(uTex2, pos.x, pos.y);
  }
}

__global__ void __advectDensity(float *d, hermes::cuda::Grid2Info dInfo,
                                hermes::cuda::Grid2Info uInfo,
                                hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * dInfo.resolution.x + x;
  if (x < dInfo.resolution.x && y < dInfo.resolution.y) {
    hermes::cuda::point2f wp = dInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f up = uInfo.toField(wp);
    hermes::cuda::point2f vp = vInfo.toField(wp);
    hermes::cuda::vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::point2f pos = dInfo.toField(wp - vel * dt);
    d[index] = tex2D(densityTex2, pos.x, pos.y);
    // d[index] = tex2D(densityTex2, x, y);
  }
}

GridSmokeSolver2::~GridSmokeSolver2() {
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(densityTex2);
}

void GridSmokeSolver2::setResolution(const ponos::uivec2 &res) {
  resolution = hermes::cuda::vec2u(res.x, res.y);
  velocity.resize(resolution);
  density.resize(resolution);
  pressure.resize(resolution);
  divergence.resize(resolution);
}

void GridSmokeSolver2::setDx(float dx) {
  velocity.setDx(dx);
  density.setDx(dx);
  pressure.setDx(dx);
  divergence.setDx(dx);
}

void GridSmokeSolver2::setOrigin(const ponos::point2f &o) {
  hermes::cuda::point2f p(o.x, o.y);
  velocity.setOrigin(p);
  density.setOrigin(p);
  pressure.setOrigin(p);
  divergence.setOrigin(p);
}

void GridSmokeSolver2::init() { setupTextures(); }

void GridSmokeSolver2::step(float dt) {
  // advectVelocities(dt);
  advectDensity(dt);
}

hermes::cuda::StaggeredGridTexture2 &GridSmokeSolver2::velocityData() {
  return velocity;
}

const hermes::cuda::GridTexture2<float> &GridSmokeSolver2::densityData() const {
  return density;
}

hermes::cuda::GridTexture2<float> &GridSmokeSolver2::densityData() {
  return density;
}

void GridSmokeSolver2::setupTextures() {
  // uTex2.addressMode[0] = cudaAddressModeBorder;
  // uTex2.addressMode[1] = cudaAddressModeBorder;
  uTex2.filterMode = cudaFilterModeLinear;
  uTex2.normalized = 0;
  // vTex2.addressMode[0] = cudaAddressModeBorder;
  // vTex2.addressMode[1] = cudaAddressModeBorder;
  vTex2.filterMode = cudaFilterModeLinear;
  vTex2.normalized = 0;
  // densityTex2.addressMode[0] = cudaAddressModeBorder;
  // densityTex2.addressMode[1] = cudaAddressModeBorder;
  densityTex2.filterMode = cudaFilterModeLinear;
  densityTex2.normalized = 0;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  using namespace hermes::cuda;
  CUDA_CHECK(cudaBindTextureToArray(
      uTex2, velocity.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vTex2, velocity.v().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      densityTex2, density.texture().textureArray(), channelDesc));
}

void GridSmokeSolver2::advectVelocities(float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(resolution.x + 1, resolution.y);
    __advectUVelocities<<<td.gridSize, td.blockSize>>>(
        velocity.uDeviceData(), velocity.u().info(), velocity.v().info(), dt);
  }
}

void GridSmokeSolver2::advectDensity(float dt) {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  __advectDensity<<<td.gridSize, td.blockSize>>>(
      density.texture().deviceData(), density.info(), velocity.u().info(),
      velocity.v().info(), dt);
  density.texture().updateTextureMemory();
}

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_KERNELS_H