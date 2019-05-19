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

#include <poseidon/math/cuda_pcg.h>
#include <poseidon/solvers/cuda_smoke_solver3_kernels.h>

namespace poseidon {

namespace cuda {

texture<float, cudaTextureType3D> uTex3, uCopyTex3;
texture<float, cudaTextureType3D> vTex3, vCopyTex3;
texture<float, cudaTextureType3D> wTex3, wCopyTex3;
texture<float, cudaTextureType3D> densityTex3;
texture<float, cudaTextureType3D> pressureTex3;
texture<float, cudaTextureType3D> divergenceTex3;
texture<unsigned char, cudaTextureType3D> solidTex3;
texture<float, cudaTextureType3D> uSolidTex3;
texture<float, cudaTextureType3D> vSolidTex3;
texture<float, cudaTextureType3D> wSolidTex3;
texture<float, cudaTextureType3D> forceTex3;
texture<float, cudaTextureType3D> temperatureTex3;

using namespace hermes::cuda;

__global__ void __injectTemperature(RegularGrid3Accessor<float> t,
                                    RegularGrid3Accessor<float> tTarget,
                                    float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (t.isIndexStored(x, y, z))
    t(x, y, z) += (1 - expf(-dt)) * (tTarget(x, y, z) - t(x, y, z));
}

template <>
void injectTemperature(
    hermes::cuda::RegularGrid3<MemoryLocation::DEVICE, float> &temperature,
    hermes::cuda::RegularGrid3<MemoryLocation::DEVICE, float>
        &targetTemperature,
    float dt) {
  hermes::ThreadArrayDistributionInfo td(temperature.resolution());
  __injectTemperature<<<td.gridSize, td.blockSize>>>(
      temperature.accessor(), targetTemperature.accessor(), dt);
}

__global__ void __injectSmoke(RegularGrid3Accessor<float> s,
                              RegularGrid3Accessor<unsigned char> source,
                              float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (s.isIndexStored(x, y, z)) {
    s(x, y, z) += dt * source(x, y, z);
    s(x, y, z) = fminf(1.f, s(x, y, z));
  }
}

template <>
void injectSmoke(
    hermes::cuda::RegularGrid3<MemoryLocation::DEVICE, float> &smoke,
    hermes::cuda::RegularGrid3<MemoryLocation::DEVICE, unsigned char> &source,
    float dt) {
  hermes::ThreadArrayDistributionInfo td(smoke.resolution());
  __injectSmoke<<<td.gridSize, td.blockSize>>>(smoke.accessor(),
                                               source.accessor(), dt);
}

__global__ void __applyForceField(RegularGrid3Accessor<float> velocity,
                                  RegularGrid3Accessor<float> force, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (velocity.isIndexStored(x, y, z)) {
    auto wp = velocity.worldPosition(x, y, z);
    auto p = force.gridPosition(wp) + vec3f(0.5f);
    velocity(x, y, z) += dt * tex3D(forceTex3, p.x, p.y, p.z);
  }
}

template <>
void applyForceField(StaggeredGrid3D &velocity, VectorGrid3D &forceField,
                     float dt) {
  Array3<float> forceArray(forceField.u().resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(forceTex3, forceArray.data(), channelDesc));
  {
    memcpy(forceArray, forceField.u().data());
    hermes::ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), forceField.u().accessor(), dt);
  }
  {
    memcpy(forceArray, forceField.v().data());
    hermes::ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), forceField.v().accessor(), dt);
  }
  {
    memcpy(forceArray, forceField.w().data());
    hermes::ThreadArrayDistributionInfo td(velocity.w().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.w().accessor(), forceField.w().accessor(), dt);
  }
  cudaUnbindTexture(forceTex3);
}

__global__ void
__applyBuoyancyForceField(RegularGrid3Accessor<float> velocity,
                          RegularGrid3Accessor<float> temperature, float tamb,
                          float alpha, float beta, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (velocity.isIndexStored(x, y, z)) {
    auto wp = velocity.worldPosition(x, y, z);
    auto p = temperature.gridPosition(wp) + vec3f(0.5f);
    velocity(x, y, z) +=
        dt * (-alpha * tex3D(densityTex3, p.x, p.y, p.z) +
              beta * (tex3D(temperatureTex3, p.x, p.y, p.z) - tamb));
  }
}

template <>
void applyBuoyancyForceField(StaggeredGrid3D &velocity, RegularGrid3Df &density,
                             RegularGrid3Df &temperature,
                             float ambientTemperature, float alpha, float beta,
                             float dt) {
  Array3<float> tArray(temperature.resolution());
  Array3<float> dArray(density.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(
      cudaBindTextureToArray(temperatureTex3, tArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(densityTex3, dArray.data(), channelDesc));
  memcpy(tArray, temperature.data());
  memcpy(dArray, density.data());
  hermes::ThreadArrayDistributionInfo td(velocity.v().resolution());
  __applyBuoyancyForceField<<<td.gridSize, td.blockSize>>>(
      velocity.v().accessor(), temperature.accessor(), ambientTemperature,
      alpha, beta, dt);
  cudaUnbindTexture(temperatureTex3);
  cudaUnbindTexture(densityTex3);
}

__global__ void __computeDivergence(RegularGrid3Accessor<float> divergence,
                                    vec3f invdx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (divergence.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    float left = tex3D(uTex3, xc, yc, zc);
    float right = tex3D(uTex3, xc + 1, yc, zc);
    float bottom = tex3D(vTex3, xc, yc, zc);
    float top = tex3D(vTex3, xc, yc + 1, zc);
    float back = tex3D(wTex3, xc, yc, zc);
    float front = tex3D(wTex3, xc, yc, zc + 1);
    unsigned char sleft = tex3D(solidTex3, xc - 1, yc, zc);
    unsigned char sright = tex3D(solidTex3, xc + 1, yc, zc);
    unsigned char sbottom = tex3D(solidTex3, xc, yc - 1, zc);
    unsigned char stop = tex3D(solidTex3, xc, yc + 1, zc);
    unsigned char sback = tex3D(solidTex3, xc, yc, zc - 1);
    unsigned char sfront = tex3D(solidTex3, xc, yc, zc + 1);
    if (sleft)
      left = 0; // tex3D(uSolidTex3, xc, yc, zc);
    if (sright)
      right = 0; // tex3D(uSolidTex3, xc + 1, yc, zc);
    if (sbottom)
      bottom = 0; // tex3D(vSolidTex3, xc, yc, zc);
    if (stop)
      top = 0; // tex3D(vSolidTex3, xc, yc + 1, zc);
    if (sback)
      back = 0; // tex3D(wSolidTex3, xc, yc, zc);
    if (sfront)
      front = 0; // tex3D(wSolidTex3, xc, yc, zc + 1);
    divergence(x, y, z) =
        dot(invdx, vec3f(right - left, top - bottom, front - back));
  }
}

template <>
void computeDivergence(
    StaggeredGrid3D &velocity,
    RegularGrid3<MemoryLocation::DEVICE, unsigned char> &solid,
    RegularGrid3Df &divergence) {
  Array3<float> uArray(velocity.u().resolution());
  Array3<float> vArray(velocity.v().resolution());
  Array3<float> wArray(velocity.w().resolution());
  Array3<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(uTex3, uArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(vTex3, vArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wTex3, wArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, sArray.data(), channelDesc));
  memcpy(uArray, velocity.u().data());
  memcpy(vArray, velocity.v().data());
  memcpy(wArray, velocity.w().data());
  memcpy(sArray, solid.data());
  auto info = divergence.info();
  vec3f inv(1.f / divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(divergence.accessor(),
                                                     inv);
  cudaUnbindTexture(uTex3);
  cudaUnbindTexture(vTex3);
  cudaUnbindTexture(wTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __fillPressureMatrix(MemoryBlock3Accessor<FDMatrix3Entry> A,
                                     RegularGrid3Accessor<unsigned char> solid,
                                     float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (A.isIndexValid(i, j, k)) {
    if (solid(i, j, k))
      return;
    A(i, j, k).diag = 0;
    A(i, j, k).x = 0;
    A(i, j, k).y = 0;
    A(i, j, k).z = 0;
    // left - right
    if (solid.isIndexStored(i - 1, j, k) && !solid(i - 1, j, k))
      A(i, j, k).diag += scale;
    if (solid.isIndexStored(i + 1, j, k)) {
      if (!solid(i + 1, j, k)) {
        A(i, j, k).diag += scale;
        A(i, j, k).x = -scale;
      } // else // EMPTY
      //   A(i, j, k).diag += scale;
    }
    // bottom - top
    if (solid.isIndexStored(i, j - 1, k) && !solid(i, j - 1, k))
      A(i, j, k).diag += scale;
    if (solid.isIndexStored(i, j + 1, k)) {
      if (!solid(i, j + 1, k)) {
        A(i, j, k).diag += scale;
        A(i, j, k).y = -scale;
      } // else // EMPTY
      //   A(i, j, k).diag += scale;
    }
    // back - front
    if (solid.isIndexStored(i, j, k - 1) && !solid(i, j, k - 1))
      A(i, j, k).diag += scale;
    if (solid.isIndexStored(i, j, k + 1)) {
      if (!solid(i, j, k + 1)) {
        A(i, j, k).diag += scale;
        A(i, j, k).z = -scale;
      } //  else // EMPTY
        //   A(i, j, k).diag += scale;
    }
  }
}

__global__ void __buildRHS(MemoryBlock3Accessor<int> indices,
                           RegularGrid3Accessor<float> divergence,
                           MemoryBlock1Accessor<float> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (indices.isIndexValid(i, j, k)) {
    if (indices(i, j, k) >= 0)
      rhs[indices(i, j, k)] = divergence(i, j, k);
  }
}

__global__ void __1To3(MemoryBlock3Accessor<int> indices,
                       MemoryBlock1Accessor<float> v,
                       MemoryBlock3Accessor<float> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (indices.isIndexValid(i, j, k)) {
    if (indices(i, j, k) >= 0)
      m(i, j, k) = v[indices(i, j, k)];
  }
}

template <>
size_t setupPressureSystem(RegularGrid3Df &divergence, RegularGrid3Duc &solid,
                           FDMatrix3D &pressureMatrix, float dt,
                           MemoryBlock1Df &rhs) {
  // fill matrix
  float scale = dt / (divergence.spacing().x * divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
      pressureMatrix.dataAccessor(), solid.accessor(), scale);
  // compute indices
  auto res = divergence.resolution();
  MemoryBlock3<MemoryLocation::HOST, int> h_indices(res);
  h_indices.allocate();
  MemoryBlock3<MemoryLocation::HOST, unsigned char> h_solid(res);
  h_solid.allocate();
  memcpy(h_solid, solid.data());
  auto solidAcc = h_solid.accessor();
  auto indicesAcc = h_indices.accessor();
  int curIndex = 0;
  for (size_t k = 0; k < res.z; k++)
    for (size_t j = 0; j < res.y; j++)
      for (size_t i = 0; i < res.x; i++)
        if (!solidAcc(i, j, k)) {
          indicesAcc(i, j, k) = curIndex++;
        } else
          indicesAcc(i, j, k) = -1;
  memcpy(pressureMatrix.indexData(), h_indices);
  // rhs
  rhs.resize(curIndex);
  rhs.allocate();
  __buildRHS<<<td.gridSize, td.blockSize>>>(pressureMatrix.indexDataAccessor(),
                                            divergence.accessor(),
                                            rhs.accessor());
  return curIndex;
}

template <>
void solvePressureSystem(
    FDMatrix3D &A, RegularGrid3Df &divergence, RegularGrid3Df &pressure,
    RegularGrid3<MemoryLocation::DEVICE, unsigned char> &solid, float dt) {
  // setup system
  MemoryBlock1Df rhs;
  setupPressureSystem(divergence, solid, A, dt, rhs);
  // apply incomplete Cholesky preconditioner
  // solve system
  MemoryBlock1Df x(rhs.size(), 0.f);
  float residual;
  pcg(x, A, rhs, 100, &residual);
  // store pressure values
  hermes::ThreadArrayDistributionInfo td(pressure.resolution());
  __1To3<<<td.gridSize, td.blockSize>>>(A.indexDataAccessor(), x.accessor(),
                                        pressure.data().accessor());
}

__global__ void __projectionStepU(RegularGrid3Accessor<float> u, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (u.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    if (tex3D(solidTex3, xc - 1, yc, zc))
      u(x, y, z) = 0; // tex3D(uSolidTex3, xc - 1, yc, zc);
    else if (tex3D(solidTex3, xc, yc, zc))
      u(x, y, z) = 0; // tex3D(uSolidTex3, xc, yc, zc);
    else {
      float l = tex3D(pressureTex3, xc - 1, yc, zc);
      float r = tex3D(pressureTex3, xc, yc, zc);
      u(x, y, z) -= scale * (r - l);
    }
  }
}

__global__ void __projectionStepV(RegularGrid3Accessor<float> v, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (v.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    if (tex3D(solidTex3, xc, yc - 1, zc))
      v(x, y, z) = 0; // tex3D(vSolidTex3, xc, yc - 1, zc);
    else if (tex3D(solidTex3, xc, yc, zc))
      v(x, y, z) = 0; // tex3D(vSolidTex3, xc, yc, zc);
    else {
      float b = tex3D(pressureTex3, xc, yc - 1, zc);
      float t = tex3D(pressureTex3, xc, yc, zc);
      v(x, y, z) -= scale * (t - b);
    }
  }
}

__global__ void __projectionStepW(RegularGrid3Accessor<float> w, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (w.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    if (tex3D(solidTex3, xc, yc, zc - 1))
      w(x, y, z) = 0; // tex3D(wSolidTex3, xc, yc, zc - 1);
    else if (tex3D(solidTex3, xc, yc, zc))
      w(x, y, z) = 0; // tex3D(wSolidTex3, xc, yc, zc);
    else {
      float b = tex3D(pressureTex3, xc, yc, zc - 1);
      float f = tex3D(pressureTex3, xc, yc, zc);
      w(x, y, z) -= scale * (f - b);
    }
  }
}

template <>
void projectionStep(RegularGrid3Df &pressure, RegularGrid3Duc &solid,
                    StaggeredGrid3D &velocity, float dt) {
  Array3<float> pArray(pressure.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(pressureTex3, pArray.data(), channelDesc));
  memcpy(pArray, pressure.data());
  {
    auto info = velocity.u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepU<<<td.gridSize, td.blockSize>>>(velocity.u().accessor(),
                                                     scale);
  }
  {
    auto info = velocity.v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepV<<<td.gridSize, td.blockSize>>>(velocity.v().accessor(),
                                                     scale);
  }
  {
    auto info = velocity.w().info();
    float invdx = 1.0 / info.spacing.z;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepU<<<td.gridSize, td.blockSize>>>(velocity.w().accessor(),
                                                     scale);
  }
  cudaUnbindTexture(pressureTex3);
}

} // namespace cuda

} // namespace poseidon
