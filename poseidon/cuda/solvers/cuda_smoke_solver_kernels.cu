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

#include <poseidon/solvers/cuda_smoke_solver_kernels.h>

namespace poseidon {

namespace cuda {

texture<float, cudaTextureType2D> uTex2, uCopyTex2;
texture<float, cudaTextureType2D> vTex2, vCopyTex2;
texture<float, cudaTextureType2D> densityTex2;
texture<float, cudaTextureType2D> pressureTex2;
texture<float, cudaTextureType2D> divergenceTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;
texture<float, cudaTextureType2D> uSolidTex2;
texture<float, cudaTextureType2D> vSolidTex2;
texture<float, cudaTextureType2D> uForceTex2;
texture<float, cudaTextureType2D> vForceTex2;

__global__ void __applyForceFieldU(float *u, hermes::cuda::Grid2Info uInfo,
                                   float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * uInfo.resolution.x + x;
  if (x < uInfo.resolution.x && y < uInfo.resolution.y) {
    u[index] += dt * tex2D(uForceTex2, x + 0.5, y + 0.5);
  }
}

__global__ void __applyForceFieldV(float *v, hermes::cuda::Grid2Info vInfo,
                                   float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    v[index] += dt * tex2D(vForceTex2, x + 0.5, y + 0.5);
  }
}

__global__ void __computeDivergenceStaggered(float *d,
                                             hermes::cuda::Grid2Info dInfo,
                                             float invdx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * dInfo.resolution.x + x;
  if (x < dInfo.resolution.x && y < dInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float l = tex2D(uTex2, xc, yc);
    float r = tex2D(uTex2, xc + 1, yc);
    float b = tex2D(vTex2, xc, yc);
    float t = tex2D(vTex2, xc, yc + 1);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    if (sl)
      l = tex2D(uSolidTex2, xc, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc + 1, yc);
    if (sb)
      b = tex2D(vSolidTex2, xc, yc);
    if (st)
      t = tex2D(vSolidTex2, xc, yc + 1);
    d[index] = invdx * (t - b + r - l);
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
    float l = tex2D(uTex2, xc - 1, yc);
    float r = tex2D(uTex2, xc + 1, yc);
    float b = tex2D(vTex2, xc, yc - 1);
    float t = tex2D(vTex2, xc, yc + 1);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    if (sl)
      l = tex2D(uSolidTex2, xc - 1, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc + 1, yc);
    if (sb)
      b = tex2D(vSolidTex2, xc, yc - 1);
    if (st)
      t = tex2D(vSolidTex2, xc, yc + 1);
    d[index] = invdx * 0.5 * (t - b + r - l);
  }
}

__global__ void __computePressure(float *p, hermes::cuda::Grid2Info pInfo,
                                  float alpha) {
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
    p[index] = (l + r + t + b + alpha * rhs) * 0.25;
  }
}

__global__ void __diffuseU(float *u, hermes::cuda::Grid2Info uInfo, float k,
                           float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * uInfo.resolution.x + x;
  if (x < uInfo.resolution.x && y < uInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float l = tex2D(uTex2, xc - 1, yc);
    float r = tex2D(uTex2, xc + 1, yc);
    float b = tex2D(uTex2, xc, yc - 1);
    float t = tex2D(uTex2, xc, yc + 1);
    float rhs = tex2D(uCopyTex2, xc, yc);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    if (sl)
      l = tex2D(uSolidTex2, xc - 1, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc, yc);
    if (sb)
      b = tex2D(uSolidTex2, xc, yc - 1);
    if (st)
      t = tex2D(uSolidTex2, xc, yc + 1);
    float scale = dt * k / (uInfo.dx * uInfo.dx);
    u[index] = (scale * (l + r + t + b) + rhs) / (1 + 4 * scale);
  }
}

__global__ void __diffuseV(float *v, hermes::cuda::Grid2Info vInfo, float k,
                           float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float l = tex2D(vTex2, xc - 1, yc);
    float r = tex2D(vTex2, xc + 1, yc);
    float b = tex2D(vTex2, xc, yc - 1);
    float t = tex2D(vTex2, xc, yc + 1);
    float rhs = tex2D(vCopyTex2, xc, yc);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc);
    if (sl)
      l = tex2D(uSolidTex2, xc - 1, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc + 1, yc);
    if (sb)
      b = tex2D(uSolidTex2, xc, yc - 1);
    if (st)
      t = tex2D(uSolidTex2, xc, yc);
    float scale = dt * k / (vInfo.dx * vInfo.dx);
    v[index] = (scale * (l + r + t + b) + rhs) / (1 + 4 * scale);
  }
}

__global__ void __diffuse(float *v, hermes::cuda::Grid2Info vInfo, float k,
                          float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * vInfo.resolution.x + x;
  if (x < vInfo.resolution.x && y < vInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float l = tex2D(vTex2, xc - 1, yc);
    float r = tex2D(vTex2, xc + 1, yc);
    float b = tex2D(vTex2, xc, yc - 1);
    float t = tex2D(vTex2, xc, yc + 1);
    float rhs = tex2D(vCopyTex2, xc, yc);
    unsigned char sl = tex2D(solidTex2, xc - 1, yc);
    unsigned char sr = tex2D(solidTex2, xc + 1, yc);
    unsigned char sb = tex2D(solidTex2, xc, yc - 1);
    unsigned char st = tex2D(solidTex2, xc, yc + 1);
    if (sl)
      l = tex2D(uSolidTex2, xc - 1, yc);
    if (sr)
      r = tex2D(uSolidTex2, xc + 1, yc);
    if (sb)
      b = tex2D(uSolidTex2, xc, yc - 1);
    if (st)
      t = tex2D(uSolidTex2, xc, yc + 1);
    float scale = dt * k / (vInfo.dx * vInfo.dx);
    v[index] = (scale * (l + r + t + b) + rhs) / (1 + 4 * scale);
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
    if (tex2D(solidTex2, xc - 1, yc))
      u[index] = tex2D(uSolidTex2, xc - 1, yc);
    else if (tex2D(solidTex2, xc, yc))
      u[index] = tex2D(uSolidTex2, xc, yc);
    else {
      float l = tex2D(pressureTex2, xc - 1, yc);
      float r = tex2D(pressureTex2, xc, yc);
      u[index] -= scale * (r - l);
    }
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
    if (tex2D(solidTex2, xc, yc - 1))
      v[index] = tex2D(vSolidTex2, xc, yc - 1);
    else if (tex2D(solidTex2, xc, yc))
      v[index] = tex2D(vSolidTex2, xc, yc);
    else {
      float b = tex2D(pressureTex2, xc, yc - 1);
      float t = tex2D(pressureTex2, xc, yc);
      v[index] -= scale * (t - b);
    }
  }
}

__global__ void __projectionStep(float *u, float *v,
                                 hermes::cuda::Grid2Info info, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * info.resolution.x + x;
  if (x < info.resolution.x && y < info.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    if (tex2D(solidTex2, xc, yc - 1))
      v[index] = tex2D(vSolidTex2, xc, yc - 1);
    else if (tex2D(solidTex2, xc, yc + 1))
      v[index] = tex2D(vSolidTex2, xc, yc + 1);
    else {
      float b = tex2D(pressureTex2, xc, yc - 1);
      float t = tex2D(pressureTex2, xc, yc + 1);
      v[index] -= scale * 0.5 * (t - b);
    }
    if (tex2D(solidTex2, xc - 1, yc))
      u[index] = tex2D(vSolidTex2, xc - 1, yc);
    else if (tex2D(solidTex2, xc + 1, yc))
      u[index] = tex2D(vSolidTex2, xc + 1, yc);
    else {
      float l = tex2D(pressureTex2, xc - 1, yc);
      float r = tex2D(pressureTex2, xc + 1, yc);
      u[index] -= scale * 0.5 * (r - l);
    }
  }
}

__global__ void __diffuseFFT(cufftComplex *wu, cufftComplex *wv, int nx, int ny,
                             float viscosity, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j > ny / 2 || i >= nx)
    return;
  int index = i * (ny / 2 + 1) + j;
  cufftComplex k;
  k.x = j;
  k.y = (i <= nx / 2) ? i : i - nx;
  float kk = k.x * k.x + k.y * k.y;
  wu[index].x /= (1.f + viscosity * dt * kk);
  wu[index].y /= (1.f + viscosity * dt * kk);
  wv[index].x /= (1.f + viscosity * dt * kk);
  wv[index].y /= (1.f + viscosity * dt * kk);
}

__device__ __host__ inline cufftComplex add(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__device__ __host__ inline cufftComplex sub(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

__device__ __host__ inline cufftComplex mul(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__device__ __host__ inline cufftComplex conj(cufftComplex a) {
  cufftComplex c;
  c.x = a.x;
  c.y = -a.y;
  return c;
}

__device__ __host__ inline cufftComplex muls(cufftComplex a, float s) {
  cufftComplex c;
  c.x = a.x * s;
  c.y = a.y * s;
  return c;
}

__global__ void __projectFFT(cufftComplex *wu, cufftComplex *wv, int nx,
                             int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ nx - 1
  int j = blockIdx.y * blockDim.y + threadIdx.y; // 0 ~ ny / 2
  if (j > ny / 2 || i >= nx)
    return;
  int index = i * (ny / 2 + 1) + j;
  cufftComplex k;
  k.x = j;
  k.y = (i <= nx / 2) ? i : i - nx;
  float kk = k.x * k.x + k.y * k.y;
  if (i == 0 && j == 0)
    return;
  wu[index].x -= k.x * k.x * wu[index].x / kk + k.y * k.x * wv[index].x / kk;
  wu[index].y -= k.x * k.x * wu[index].y / kk + k.y * k.x * wv[index].y / kk;
  wv[index].x -= k.x * k.y * wu[index].x / kk + k.y * k.y * wv[index].x / kk;
  wv[index].y -= k.x * k.y * wu[index].y / kk + k.y * k.y * wv[index].y / kk;
  // cufftComplex I, J;
  // I.x = i;
  // I.y = 0;
  // J.x = j;
  // J.y = 0;
  // [ir + 0img, jr + 0img] [wu.xr + wu.yimg, wv.xr + wv.yr]
  // cufftComplex dotKW = add(mul(I, conj(wu[index])), mul(J, conj(wv[index])));
  // wu[index] = sub(wu[index], muls(mul(dotKW, I), 1.f / (i * i + j * j)));
  // wv[index] = sub(wv[index], muls(mul(dotKW, J), 1.f / (i * i + j * j)));
}

void unbindTextures() {
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(densityTex2);
  cudaUnbindTexture(divergenceTex2);
  cudaUnbindTexture(pressureTex2);
  cudaUnbindTexture(solidTex2);
  cudaUnbindTexture(uSolidTex2);
  cudaUnbindTexture(vSolidTex2);
  cudaUnbindTexture(uForceTex2);
  cudaUnbindTexture(vForceTex2);
}

void bindTextures(const hermes::cuda::VectorGridTexture2 &velocity,
                  const hermes::cuda::VectorGridTexture2 &velocityCopy,
                  const hermes::cuda::GridTexture2<float> &density,
                  const hermes::cuda::GridTexture2<float> &divergence,
                  const hermes::cuda::GridTexture2<float> &pressure,
                  const hermes::cuda::GridTexture2<unsigned char> &solid,
                  const hermes::cuda::VectorGridTexture2 &forceField,
                  const hermes::cuda::VectorGridTexture2 &solidVelocity) {
  using namespace hermes::cuda;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(
      uCopyTex2, velocityCopy.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vCopyTex2, velocityCopy.v().texture().textureArray(), channelDesc));
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
  CUDA_CHECK(cudaBindTextureToArray(
      uForceTex2, forceField.u().texture().textureArray(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(
      vForceTex2, forceField.v().texture().textureArray(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, solid.texture().textureArray(),
                                    channelDesc));
}

void setupTextures() {
  // uTex2.addressMode[0] = cudaAddressModeBorder;
  // uTex2.addressMode[1] = cudaAddressModeBorder;
  uSolidTex2.filterMode = cudaFilterModePoint;
  uSolidTex2.normalized = 0;
  vSolidTex2.filterMode = cudaFilterModePoint;
  vSolidTex2.normalized = 0;
  uForceTex2.filterMode = cudaFilterModePoint;
  uForceTex2.normalized = 0;
  vForceTex2.filterMode = cudaFilterModePoint;
  vForceTex2.normalized = 0;
  uTex2.filterMode = cudaFilterModeLinear;
  uTex2.normalized = 0;
  vTex2.filterMode = cudaFilterModeLinear;
  vTex2.normalized = 0;
  uCopyTex2.filterMode = cudaFilterModeLinear;
  uCopyTex2.normalized = 0;
  vCopyTex2.filterMode = cudaFilterModeLinear;
  vCopyTex2.normalized = 0;
  densityTex2.filterMode = cudaFilterModeLinear;
  densityTex2.normalized = 0;
  divergenceTex2.filterMode = cudaFilterModePoint;
  divergenceTex2.normalized = 0;
  pressureTex2.filterMode = cudaFilterModePoint;
  pressureTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
}

void applyForceField(hermes::cuda::VectorGridTexture2 &velocity,
                     const hermes::cuda::VectorGridTexture2 &forceField,
                     float dt) {
  {
    auto info = velocity.v().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __applyForceFieldV<<<td.gridSize, td.blockSize>>>(velocity.vDeviceData(),
                                                      velocity.v().info(), dt);
    velocity.v().texture().updateTextureMemory();
  }
  {
    auto info = velocity.u().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __applyForceFieldU<<<td.gridSize, td.blockSize>>>(velocity.uDeviceData(),
                                                      velocity.u().info(), dt);
    velocity.u().texture().updateTextureMemory();
  }
}

void computeDivergence(const hermes::cuda::StaggeredGridTexture2 &velocity,
                       const hermes::cuda::GridTexture2<unsigned char> &solid,
                       hermes::cuda::GridTexture2<float> &divergence) {
  auto info = divergence.info();
  float invdx = 1.0 / info.dx;
  hermes::ThreadArrayDistributionInfo td(info.resolution);
  __computeDivergenceStaggered<<<td.gridSize, td.blockSize>>>(
      divergence.texture().deviceData(), divergence.info(), invdx);
  divergence.texture().updateTextureMemory();
}

void computeDivergence(const hermes::cuda::VectorGridTexture2 &velocity,
                       const hermes::cuda::GridTexture2<unsigned char> &solid,
                       hermes::cuda::GridTexture2<float> &divergence) {
  auto info = divergence.info();
  float invdx = 1.0 / info.dx;
  hermes::ThreadArrayDistributionInfo td(info.resolution);
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      divergence.texture().deviceData(), divergence.info(), invdx);
  divergence.texture().updateTextureMemory();
}

void computePressure(const hermes::cuda::GridTexture2<float> &divergence,
                     const hermes::cuda::GridTexture2<unsigned char> &solid,
                     hermes::cuda::GridTexture2<float> &pressure, float dt,
                     int iterations) {
  auto info = pressure.info();
  hermes::ThreadArrayDistributionInfo td(info.resolution);
  float alpha = -(info.dx * info.dx) / dt;
  for (int i = 0; i < iterations; i++) {
    pressure.texture().updateTextureMemory();
    __computePressure<<<td.gridSize, td.blockSize>>>(
        pressure.texture().deviceData(), pressure.info(), alpha);
    using namespace hermes::cuda;
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  pressure.texture().updateTextureMemory();
}

void diffuse(hermes::cuda::VectorGridTexture2 &velocity, float k, float dt,
             int iterations) {
  {
    auto info = velocity.u().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    for (int i = 0; i < iterations; i++) {
      velocity.u().texture().updateTextureMemory();
      __diffuse<<<td.gridSize, td.blockSize>>>(velocity.uDeviceData(), info, k,
                                               dt);
    }
  }
  {
    auto info = velocity.v().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    for (int i = 0; i < iterations; i++) {
      velocity.v().texture().updateTextureMemory();
      __diffuse<<<td.gridSize, td.blockSize>>>(velocity.vDeviceData(), info, k,
                                               dt);
    }
  }
  velocity.u().texture().updateTextureMemory();
  velocity.v().texture().updateTextureMemory();
}

void diffuse(hermes::cuda::StaggeredGridTexture2 &velocity, float k, float dt,
             int iterations) {
  {
    auto info = velocity.u().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    for (int i = 0; i < iterations; i++) {
      velocity.u().texture().updateTextureMemory();
      __diffuseU<<<td.gridSize, td.blockSize>>>(velocity.uDeviceData(), info, k,
                                                dt);
    }
  }
  {
    auto info = velocity.v().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    for (int i = 0; i < iterations; i++) {
      velocity.v().texture().updateTextureMemory();
      __diffuseV<<<td.gridSize, td.blockSize>>>(velocity.vDeviceData(), info, k,
                                                dt);
    }
  }
  velocity.u().texture().updateTextureMemory();
  velocity.v().texture().updateTextureMemory();
}

void projectionStep(const hermes::cuda::GridTexture2<float> &pressure,
                    const hermes::cuda::GridTexture2<unsigned char> &solid,
                    hermes::cuda::StaggeredGridTexture2 &velocity, float dt) {
  {
    auto info = velocity.u().info();
    float invdx = 1.0 / info.dx;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepU<<<td.gridSize, td.blockSize>>>(
        velocity.uDeviceData(), velocity.u().info(), scale);
  }
  {
    auto info = velocity.v().info();
    float invdx = 1.0 / info.dx;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepV<<<td.gridSize, td.blockSize>>>(
        velocity.vDeviceData(), velocity.v().info(), scale);
  }
  velocity.u().texture().updateTextureMemory();
  velocity.v().texture().updateTextureMemory();
}

void projectionStep(const hermes::cuda::GridTexture2<float> &pressure,
                    const hermes::cuda::GridTexture2<unsigned char> &solid,
                    hermes::cuda::VectorGridTexture2 &velocity, float dt) {
  auto info = velocity.v().info();
  float invdx = 1.0 / info.dx;
  float scale = dt * invdx;
  hermes::ThreadArrayDistributionInfo td(info.resolution);
  __projectionStep<<<td.gridSize, td.blockSize>>>(velocity.uDeviceData(),
                                                  velocity.vDeviceData(),
                                                  velocity.v().info(), scale);
  velocity.u().texture().updateTextureMemory();
  velocity.v().texture().updateTextureMemory();
}

void diffuseFFT(hermes::cuda::vec2u resolution, cufftComplex *d_frequenciesU,
                cufftComplex *d_frequenciesV, float k, float dt) {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y / 2 + 1);
  __diffuseFFT<<<td.gridSize, td.blockSize>>>(
      d_frequenciesU, d_frequenciesV, resolution.x, resolution.y, k, dt);
}

void projectFFT(hermes::cuda::vec2u resolution, cufftComplex *d_frequenciesU,
                cufftComplex *d_frequenciesV) {
  hermes::ThreadArrayDistributionInfo td(resolution.x, resolution.y / 2 + 1);
  __projectFFT<<<td.gridSize, td.blockSize>>>(d_frequenciesU, d_frequenciesV,
                                              resolution.x, resolution.y);
}

} // namespace cuda

} // namespace poseidon
