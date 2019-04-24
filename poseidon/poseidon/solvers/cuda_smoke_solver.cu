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
texture<unsigned char, cudaTextureType2D> solidTex2;
texture<float, cudaTextureType2D> uSolidTex2;
texture<float, cudaTextureType2D> vSolidTex2;

__global__ void __setupScene(poseidon::cuda::Collider2<float> **solids,
                             poseidon::cuda::Collider2<float> **scene) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    solids[0] = new poseidon::cuda::SphereCollider2<float>(
        hermes::cuda::point2(0.f, 0.f), 0.1f);
    float d = 1.0 / 64;
    // floor
    solids[1] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(1.f, d)));
    // ceil
    solids[2] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 1.f - d), hermes::cuda::point2(1.f, 1.f)));
    // left
    solids[3] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(d, 1.f)));
    // right
    solids[4] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(1.f - d, 0.f), hermes::cuda::point2(1.f, 1.f)));
    *scene = new poseidon::cuda::Collider2Set<float>(solids, 5);
  }
}

__global__ void __freeScene(poseidon::cuda::Collider2<float> **solids) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < 5; ++i)
      delete solids[i];
}

__global__ void __rasterColliders(Collider2<float> *const *colliders,
                                  unsigned char *solids, float *u, float *v,
                                  hermes::cuda::Grid2Info sInfo) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * sInfo.resolution.x + x;
  if (x < sInfo.resolution.x && y < sInfo.resolution.y) {
    if ((*colliders)->intersect(sInfo.toWorld(hermes::cuda::point2(x, y))))
      solids[index] = 1;
    else
      solids[index] = 0;
    u[y * sInfo.resolution.x + x] = u[y * sInfo.resolution.x + x + 1] = 0;
    v[y * sInfo.resolution.x + x] = v[(y + 1) * sInfo.resolution.x + x] = 0;
  }
}

__global__ void __addGravityV(float *v, hermes::cuda::Grid2Info vInfo, float g,
                              float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    v[index] += dt * g;
  }
}

__global__ void __advectUVelocities(float *u, hermes::cuda::Grid2Info uInfo,
                                    hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * uInfo.resolution.x + x;
  if (x < uInfo.resolution.x && y < uInfo.resolution.y) {
    unsigned char solid = tex2D(solidTex2, x + 0.5, y + 0.5);
    if (solid) {
      u[index] = 0;
      return;
    }
    hermes::cuda::point2f wp = uInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f vp = vInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::vec2f vel(tex2D(uTex2, x + 0.5, y + 0.5),
                            tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::point2f pos = wp - vel * dt + hermes::cuda::vec2(0.5);
    u[index] = tex2D(uTex2, pos.x, pos.y);
  }
}

__global__ void __advectVVelocities(float *v, hermes::cuda::Grid2Info uInfo,
                                    hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    unsigned char solid = tex2D(solidTex2, x + 0.5, y + 0.5);
    if (solid) {
      v[index] = 0;
      return;
    }
    hermes::cuda::point2f wp = vInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f up = uInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::vec2f vel(tex2D(uTex2, up.x, up.y),
                            tex2D(vTex2, x + 0.5, y + 0.5));
    hermes::cuda::point2f pos = wp - vel * dt + hermes::cuda::vec2(0.5);
    v[index] = tex2D(vTex2, pos.x, pos.y);
  }
}

__global__ void __advectDensity(float *d, hermes::cuda::Grid2Info dInfo,
                                hermes::cuda::Grid2Info uInfo,
                                hermes::cuda::Grid2Info vInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * dInfo.resolution.x + x;
  if (x < dInfo.resolution.x && y < dInfo.resolution.y) {
    unsigned char solid = tex2D(solidTex2, x + 0.5, y + 0.5);
    if (solid) {
      d[index] = 0;
      return;
    }
    hermes::cuda::point2f wp = dInfo.toWorld(hermes::cuda::point2f(x, y));
    hermes::cuda::point2f up = uInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::point2f vp = vInfo.toField(wp) + hermes::cuda::vec2(0.5);
    hermes::cuda::vec2f vel(tex2D(uTex2, up.x, up.y), tex2D(vTex2, vp.x, vp.y));
    hermes::cuda::point2f pos =
        dInfo.toField(wp - vel * dt) + hermes::cuda::vec2(0.5);
    d[index] = tex2D(densityTex2, pos.x, pos.y);
  }
}

__global__ void __computeDivergence(float *d, hermes::cuda::Grid2Info dInfo,
                                    float invdx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * dInfo.resolution.x + x;
  if (x < dInfo.resolution.x && y < dInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    float l = tex2D(uTex2, xc, yc);
    float r = tex2D(uTex2, xc + 1, yc);
    float b = tex2D(vTex2, xc, yc);
    float t = tex2D(vTex2, xc, yc + 1);
    if (sl)
      l = tex2D(uSolidTex2, xc, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc + 1, yc);
    if (sb)
      b = tex2D(vSolidTex2, xc, yc);
    if (st)
      t = tex2D(vSolidTex2, xc, yc + 1);

    d[index] = 0.5 * invdx * (t - b + r - l);
  }
}

__global__ void __computePressure(float *p, hermes::cuda::Grid2Info pInfo,
                                  float alpha, float beta) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * pInfo.resolution.x + x;
  if (x < pInfo.resolution.x && y < pInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float c = tex2D(pressureTex2, xc, yc);
    float l = tex2D(pressureTex2, xc - 1, yc);
    float r = tex2D(pressureTex2, xc + 1, yc);
    float b = tex2D(pressureTex2, xc, yc - 1);
    float t = tex2D(pressureTex2, xc, yc + 1);
    float rhs = tex2D(divergenceTex2, xc, yc);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    if (sl)
      l = c;
    if (sr)
      r = c;
    if (sb)
      b = c;
    if (st)
      t = c;
    p[index] = (l + r + t + b + alpha * rhs) * beta;
  }
}

__global__ void __projectionStepU(float *u, hermes::cuda::Grid2Info uInfo,
                                  float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * uInfo.resolution.x + x;
  if (x < uInfo.resolution.x && y < uInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float l = tex2D(pressureTex2, xc - 1, yc);
    float r = tex2D(pressureTex2, xc, yc);
    u[index] -= 0.5 * scale * (r - l);
  }
}

__global__ void __projectionStepV(float *v, hermes::cuda::Grid2Info vInfo,
                                  float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float b = tex2D(pressureTex2, xc, yc - 1);
    float t = tex2D(pressureTex2, xc, yc);
    v[index] -= 0.5 * scale * (t - b);
  }
}

GridSmokeSolver2::~GridSmokeSolver2() {
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(densityTex2);
  cudaUnbindTexture(divergenceTex2);
  cudaUnbindTexture(pressureTex2);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(uSolidTex2);
  cudaUnbindTexture(vSolidTex2);

  __freeScene<<<1, 1>>>(scene.list);
  using namespace hermes::cuda;
  CUDA_CHECK(cudaFree(scene.list));
  CUDA_CHECK(cudaFree(scene.colliders));
}

void GridSmokeSolver2::setResolution(const ponos::uivec2 &res) {
  resolution = hermes::cuda::vec2u(res.x, res.y);
  velocity.resize(resolution);
  density.resize(resolution);
  pressure.resize(resolution);
  divergence.resize(resolution);
  solid.resize(resolution);
  solidVelocity.resize(resolution);
}

void GridSmokeSolver2::setDx(float _dx) {
  dx = _dx;
  velocity.setDx(dx);
  density.setDx(dx);
  pressure.setDx(dx);
  divergence.setDx(dx);
  solid.setDx(dx);
  solidVelocity.setDx(dx);
}

void GridSmokeSolver2::setOrigin(const ponos::point2f &o) {
  hermes::cuda::point2f p(o.x, o.y);
  velocity.setOrigin(p);
  density.setOrigin(p);
  pressure.setOrigin(p);
  divergence.setOrigin(p);
  solid.setOrigin(p);
  solidVelocity.setOrigin(p);
}

void GridSmokeSolver2::init() {
  setupTextures();
  {
    using namespace hermes::cuda;
    CUDA_CHECK(cudaMalloc(&scene.list, 5 * sizeof(Collider2<float> *)));
    CUDA_CHECK(cudaMalloc(&scene.colliders, sizeof(Collider2<float> *)));
  }
}

void GridSmokeSolver2::step(float dt) {
  rasterColliders();
  advectVelocities(dt);
  advectDensity(dt);
  // inject
  addGravity(dt);
  computeDivergence();
  computePressure();
  projectionStep(dt);
}

void GridSmokeSolver2::rasterColliders() {
  __setupScene<<<1, 1>>>(scene.list, scene.colliders);
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  __rasterColliders<<<td.gridSize, td.blockSize>>>(
      scene.colliders, solid.texture().deviceData(),
      solidVelocity.uDeviceData(), solidVelocity.vDeviceData(), solid.info());
  solid.texture().updateTextureMemory();
}

void GridSmokeSolver2::setupTextures() {
  // uTex2.addressMode[0] = cudaAddressModeBorder;
  // uTex2.addressMode[1] = cudaAddressModeBorder;
  uSolidTex2.filterMode = cudaFilterModeLinear;
  uSolidTex2.normalized = 0;
  vSolidTex2.filterMode = cudaFilterModeLinear;
  vSolidTex2.normalized = 0;
  uTex2.filterMode = cudaFilterModeLinear;
  uTex2.normalized = 0;
  vTex2.filterMode = cudaFilterModeLinear;
  vTex2.normalized = 0;
  densityTex2.filterMode = cudaFilterModeLinear;
  densityTex2.normalized = 0;
  divergenceTex2.filterMode = cudaFilterModeLinear;
  divergenceTex2.normalized = 0;
  pressureTex2.filterMode = cudaFilterModeLinear;
  pressureTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  using namespace hermes::cuda;
  CUDA_CHECK(cudaBindTextureToArray(
      uTex2, velocity.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vTex2, velocity.v().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      densityTex2, density.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      divergenceTex2, divergence.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      pressureTex2, pressure.texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      uSolidTex2, solidVelocity.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vSolidTex2, solidVelocity.v().texture().textureArray(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, solid.texture().textureArray(),
                                    channelDesc));
}

void GridSmokeSolver2::addGravity(float dt) {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y + 1);
  __addGravityV<<<td.gridSize, td.blockSize>>>(velocity.vDeviceData(),
                                               velocity.v().info(), -9.81f, dt);
  velocity.v().texture().updateTextureMemory();
}

void GridSmokeSolver2::advectVelocities(float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(resolution.x + 1, resolution.y);
    __advectUVelocities<<<td.gridSize, td.blockSize>>>(
        velocity.uDeviceData(), velocity.u().info(), velocity.v().info(), dt);
  }
  {
    hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y + 1);
    __advectVVelocities<<<td.gridSize, td.blockSize>>>(
        velocity.vDeviceData(), velocity.u().info(), velocity.v().info(), dt);
  }
}

void GridSmokeSolver2::advectDensity(float dt) {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  __advectDensity<<<td.gridSize, td.blockSize>>>(
      density.texture().deviceData(), density.info(), velocity.u().info(),
      velocity.v().info(), dt);
  density.texture().updateTextureMemory();
}

void GridSmokeSolver2::computeDivergence() {
  float invdx = 1.0 / dx;
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      divergence.texture().deviceData(), divergence.info(), invdx);
  divergence.texture().updateTextureMemory();
}

void GridSmokeSolver2::computePressure() {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  float alpha = -1.0 / (dx * dx);
  float beta = 0.25;
  int iterations = 100;
  for (int i = 0; i < iterations; i++) {
    pressure.texture().updateTextureMemory();
    __computePressure<<<td.gridSize, td.blockSize>>>(
        pressure.texture().deviceData(), pressure.info(), alpha, beta);
  }
  pressure.texture().updateTextureMemory();
}

void GridSmokeSolver2::projectionStep(float dt) {
  float invdx = 1.0 / dx;
  float scale = dt * invdx;
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y);
  __projectionStepU<<<td.gridSize, td.blockSize>>>(velocity.uDeviceData(),
                                                   velocity.u().info(), scale);
  velocity.u().texture().updateTextureMemory();
  __projectionStepV<<<td.gridSize, td.blockSize>>>(velocity.vDeviceData(),
                                                   velocity.v().info(), scale);
  velocity.v().texture().updateTextureMemory();
}

hermes::cuda::StaggeredGridTexture2 &GridSmokeSolver2::velocityData() {
  return velocity;
}

const hermes::cuda::GridTexture2<float> &GridSmokeSolver2::densityData() const {
  return density;
}

const hermes::cuda::GridTexture2<unsigned char> &
GridSmokeSolver2::solidData() const {
  return solid;
}

hermes::cuda::GridTexture2<unsigned char> &GridSmokeSolver2::solidData() {
  return solid;
}

hermes::cuda::GridTexture2<float> &GridSmokeSolver2::densityData() {
  return density;
}

const hermes::cuda::StaggeredGridTexture2 &
GridSmokeSolver2::solidVelocityData() const {
  return solidVelocity;
}

hermes::cuda::StaggeredGridTexture2 &GridSmokeSolver2::solidVelocityData() {
  return solidVelocity;
}

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_KERNELS_H