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
#include <poseidon/solvers/cuda_smoke_solver2_steps.h>

namespace poseidon {

namespace cuda {

texture<float, cudaTextureType2D> uTex2, uCopyTex2, wuTex2;
texture<float, cudaTextureType2D> vTex2, vCopyTex2, wvTex2;
texture<float, cudaTextureType2D> densityTex2;
texture<float, cudaTextureType2D> pressureTex2;
texture<float, cudaTextureType2D> divergenceTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;
texture<float, cudaTextureType2D> uSolidTex2;
texture<float, cudaTextureType2D> vSolidTex2;
texture<float, cudaTextureType2D> wSolidTex2;
texture<float, cudaTextureType2D> forceTex2;
texture<float, cudaTextureType2D> temperatureTex2;

using namespace hermes::cuda;

__global__ void __injectTemperature(RegularGrid2Accessor<float> t,
                                    RegularGrid2Accessor<float> tTarget,
                                    float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (t.isIndexStored(x, y))
    t(x, y) += (1 - expf(-dt)) * (tTarget(x, y) - t(x, y));
}

template <>
void injectTemperature(
    hermes::cuda::RegularGrid2<MemoryLocation::DEVICE, float> &temperature,
    hermes::cuda::RegularGrid2<MemoryLocation::DEVICE, float>
        &targetTemperature,
    float dt) {
  hermes::ThreadArrayDistributionInfo td(temperature.resolution());
  __injectTemperature<<<td.gridSize, td.blockSize>>>(
      temperature.accessor(), targetTemperature.accessor(), dt);
}

__global__ void __injectSmoke(RegularGrid2Accessor<float> s,
                              RegularGrid2Accessor<unsigned char> source,
                              float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (s.isIndexStored(x, y)) {
    s(x, y) += dt * source(x, y) * 0.5;
    s(x, y) = fminf(1.f, s(x, y));
  }
}

template <>
void injectSmoke(
    hermes::cuda::RegularGrid2<MemoryLocation::DEVICE, float> &smoke,
    hermes::cuda::RegularGrid2<MemoryLocation::DEVICE, unsigned char> &source,
    float dt) {
  hermes::ThreadArrayDistributionInfo td(smoke.resolution());
  __injectSmoke<<<td.gridSize, td.blockSize>>>(smoke.accessor(),
                                               source.accessor(), dt);
}

__global__ void __applyForceField_t(RegularGrid2Accessor<float> velocity,
                                    RegularGrid2Accessor<float> force,
                                    float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (velocity.isIndexStored(x, y)) {
    auto wp = velocity.worldPosition(x, y);
    auto p = force.gridPosition(wp) + vec2f(0.5f);
    // printf("%f ", tex3D(forceTex3, p.x, p.y, p.z));
    velocity(x, y) += dt * tex2D(forceTex2, p.x, p.y);
  }
}

template <>
void applyForceField_t(StaggeredGrid2D &velocity, VectorGrid2D &forceField,
                       float dt) {
  forceTex2.filterMode = cudaFilterModeLinear;
  forceTex2.normalized = 0;
  Array2<float> forceArray(forceField.u().resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(forceTex2, forceArray.data(), channelDesc));
  {
    memcpy(forceArray, forceField.u().data());
    hermes::ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField_t<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), forceField.u().accessor(), dt);
  }
  {
    memcpy(forceArray, forceField.v().data());
    hermes::ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField_t<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), forceField.v().accessor(), dt);
  }
  cudaUnbindTexture(forceTex2);
}

__global__ void __applyForceField(RegularGrid2Accessor<float> velocity,
                                  RegularGrid2Accessor<unsigned char> solid,
                                  RegularGrid2Accessor<float> force, float dt,
                                  vec2u d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (velocity.isIndexStored(i, j) && !solid(i, j) &&
      !solid(i - d.x, j - d.y)) {
    velocity(i, j) += dt * (force(i - d.x, j - d.y) + force(i, j)) * 0.5f;
  }
}

template <>
void applyForceField(StaggeredGrid2D &velocity, RegularGrid2Duc &solid,
                     VectorGrid2D &forceField, float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), solid.accessor(), forceField.u().accessor(),
        dt, vec2u(1, 0));
  }
  {
    hermes::ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), solid.accessor(), forceField.v().accessor(),
        dt, vec2u(0, 1));
  }
}

__global__ void
__applyBuoyancyForceField_t(RegularGrid2Accessor<float> fu,
                            RegularGrid2Accessor<float> fv,
                            RegularGrid2Accessor<float> density,
                            RegularGrid2Accessor<float> temperature, float tamb,
                            float alpha, float beta) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (fu.isIndexStored(x, y)) {
    auto wp = fu.worldPosition(x, y);
    auto p = temperature.gridPosition(wp) + vec2f(0.5f);
    fu(x, y) = 0.f;
    fv(x, y) += -alpha * tex2D(densityTex2, p.x, p.y) +
                beta * (tex2D(temperatureTex2, p.x, p.y) - tamb);
  }
}

template <>
void computeBuoyancyForceField_t(VectorGrid2D &forceField,
                                 RegularGrid2Df &density,
                                 RegularGrid2Df &temperature,
                                 float ambientTemperature, float alpha,
                                 float beta) {
  temperatureTex2.filterMode = cudaFilterModeLinear;
  temperatureTex2.normalized = 0;
  densityTex2.filterMode = cudaFilterModeLinear;
  densityTex2.normalized = 0;
  Array2<float> tArray(temperature.resolution());
  Array2<float> dArray(density.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(
      cudaBindTextureToArray(temperatureTex2, tArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(densityTex2, dArray.data(), channelDesc));
  memcpy(tArray, temperature.data());
  memcpy(dArray, density.data());
  hermes::ThreadArrayDistributionInfo td(forceField.resolution());
  __applyBuoyancyForceField_t<<<td.gridSize, td.blockSize>>>(
      forceField.u().accessor(), forceField.v().accessor(), density.accessor(),
      temperature.accessor(), ambientTemperature, alpha, beta);
  cudaUnbindTexture(temperatureTex2);
  cudaUnbindTexture(densityTex2);
}

__global__ void __addBuoyancyForce(VectorGrid2Accessor f,
                                   RegularGrid2Accessor<unsigned char> solid,
                                   RegularGrid2Accessor<float> density,
                                   RegularGrid2Accessor<float> temperature,
                                   float tamb, float alpha, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (f.vAccessor().isIndexStored(i, j) && !solid(i, j))
    f.v(i, j) +=
        9.81 * (-alpha * density(i, j) + beta * (temperature(i, j) - tamb));
}

template <>
void addBuoyancyForce(VectorGrid2D &forceField, RegularGrid2Duc &solid,
                      RegularGrid2Df &density, RegularGrid2Df &temperature,
                      float ambientTemperature, float alpha, float beta) {
  hermes::ThreadArrayDistributionInfo td(forceField.resolution());
  __addBuoyancyForce<<<td.gridSize, td.blockSize>>>(
      forceField.accessor(), solid.accessor(), density.accessor(),
      temperature.accessor(), ambientTemperature, alpha, beta);
}

__global__ void __computeDivergence_t(RegularGrid2Accessor<float> divergence,
                                      vec2f invdx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (divergence.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float left = tex2D(uTex2, xc, yc);
    float right = tex2D(uTex2, xc + 1, yc);
    float bottom = tex2D(vTex2, xc, yc);
    float top = tex2D(vTex2, xc, yc + 1);
    unsigned char sleft = tex2D(solidTex2, xc - 1, yc);
    unsigned char sright = tex2D(solidTex2, xc + 1, yc);
    unsigned char sbottom = tex2D(solidTex2, xc, yc - 1);
    unsigned char stop = tex2D(solidTex2, xc, yc + 1);
    if (sleft)
      left = 0; // tex3D(uSolidTex3, xc, yc);
    if (sright)
      right = 0; // tex3D(uSolidTex3, xc + 1, yc, zc);
    if (sbottom)
      bottom = 0; // tex3D(vSolidTex3, xc, yc, zc);
    if (stop)
      top = 0; // tex3D(vSolidTex3, xc, yc + 1, zc);
    divergence(x, y) = dot(invdx, vec2f(right - left, top - bottom));
  }
}

template <>
void computeDivergence_t(
    StaggeredGrid2D &velocity,
    RegularGrid2<MemoryLocation::DEVICE, unsigned char> &solid,
    RegularGrid2Df &divergence) {
  uTex2.filterMode = cudaFilterModePoint;
  uTex2.normalized = 0;
  vTex2.filterMode = cudaFilterModePoint;
  vTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
  Array2<float> uArray(velocity.u().resolution());
  Array2<float> vArray(velocity.v().resolution());
  Array2<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(uTex2, uArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(vTex2, vArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, sArray.data(), channelDesc));
  memcpy(uArray, velocity.u().data());
  memcpy(vArray, velocity.v().data());
  memcpy(sArray, solid.data());
  auto info = divergence.info();
  vec2f inv(-1.f / divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence_t<<<td.gridSize, td.blockSize>>>(divergence.accessor(),
                                                       inv);
  cudaUnbindTexture(uTex2);
  cudaUnbindTexture(vTex2);
  cudaUnbindTexture(solidTex2);
}

__global__ void __computeDivergence(StaggeredGrid2Accessor vel,
                                    RegularGrid2Accessor<unsigned char> solid,
                                    StaggeredGrid2Accessor svel,
                                    RegularGrid2Accessor<float> divergence,
                                    vec2f invdx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (divergence.isIndexStored(i, j)) {
    if (solid(i, j)) {
      divergence(i, j) = 0;
      return;
    }
    float left = vel.u(i, j);
    float right = vel.u(i + 1, j);
    float bottom = vel.v(i, j);
    float top = vel.v(i, j + 1);
    unsigned char sleft = solid(i - 1, j);
    unsigned char sright = solid(i + 1, j);
    unsigned char sbottom = solid(i, j - 1);
    unsigned char stop = solid(i, j + 1);
    if (sleft)
      left = svel.u(i, j);
    if (sright)
      right = svel.u(i + 1, j);
    if (sbottom)
      bottom = svel.v(i, j);
    if (stop)
      top = svel.v(i, j + 1);
    divergence(i, j) = dot(invdx, vec2f(right - left, top - bottom));
  }
}

template <>
void computeDivergence(StaggeredGrid2D &velocity, RegularGrid2Duc &solid,
                       StaggeredGrid2D &solidVelocity,
                       RegularGrid2Df &divergence) {
  auto info = divergence.info();
  vec2f inv(-1.f / divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solidVelocity.accessor(),
      divergence.accessor(), inv);
}

__global__ void __fillPressureMatrix(MemoryBlock2Accessor<FDMatrix2Entry> A,
                                     RegularGrid2Accessor<unsigned char> solid,
                                     float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (A.isIndexValid(i, j)) {
    if (solid(i, j))
      return;
    A(i, j).diag = 0;
    A(i, j).x = 0;
    A(i, j).y = 0;
    // left - right
    if (solid.isIndexStored(i - 1, j) && !solid(i - 1, j))
      A(i, j).diag += scale;
    if (solid.isIndexStored(i + 1, j)) {
      if (!solid(i + 1, j)) {
        A(i, j).diag += scale;
        A(i, j).x = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(i, j).diag += scale;
    // bottom - top
    if (solid.isIndexStored(i, j - 1) && !solid(i, j - 1))
      A(i, j).diag += scale;
    if (solid.isIndexStored(i, j + 1)) {
      if (!solid(i, j + 1)) {
        A(i, j).diag += scale;
        A(i, j).y = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(i, j).diag += scale;
  }
}

__global__ void __buildRHS(MemoryBlock2Accessor<int> indices,
                           RegularGrid2Accessor<float> divergence,
                           MemoryBlock1Accessor<double> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      rhs[indices(i, j)] = divergence(i, j);
  }
}

__global__ void __1To2(MemoryBlock2Accessor<int> indices,
                       MemoryBlock1Accessor<double> v,
                       MemoryBlock2Accessor<float> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      m(i, j) = v[indices(i, j)];
  }
}

template <>
size_t setupPressureSystem(RegularGrid2Df &divergence, RegularGrid2Duc &solid,
                           FDMatrix2D &pressureMatrix, float dt,
                           MemoryBlock1Dd &rhs) {
  // fill matrix
  float scale = dt / (divergence.spacing().x * divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
      pressureMatrix.dataAccessor(), solid.accessor(), scale);
  // compute indices
  auto res = divergence.resolution();
  MemoryBlock2<MemoryLocation::HOST, int> h_indices(res);
  h_indices.allocate();
  MemoryBlock2<MemoryLocation::HOST, unsigned char> h_solid(res);
  h_solid.allocate();
  memcpy(h_solid, solid.data());
  auto solidAcc = h_solid.accessor();
  auto indicesAcc = h_indices.accessor();
  int curIndex = 0;
  for (size_t j = 0; j < res.y; j++)
    for (size_t i = 0; i < res.x; i++)
      if (!solidAcc(i, j)) {
        indicesAcc(i, j) = curIndex++;
      } else
        indicesAcc(i, j) = -1;
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
void solvePressureSystem(FDMatrix2D &A, RegularGrid2Df &divergence,
                         RegularGrid2Df &pressure, RegularGrid2Duc &solid,
                         float dt) {
  // setup system
  MemoryBlock1Dd rhs;
  setupPressureSystem(divergence, solid, A, dt, rhs);
  // apply incomplete Cholesky preconditioner
  // solve system
  MemoryBlock1Dd x(rhs.size(), 0.f);
  // FDMatrix3H H(A.gridSize());
  // H.copy(A);
  // auto acc = H.accessor();
  // std::cerr << acc << "rhs\n" << rhs << std::endl;
  std::cerr << "solve\n";
  pcg(x, A, rhs, rhs.size(), 1e-12);
  // std::cerr << residual << "\n" << x << std::endl;
  // store pressure values
  MemoryBlock1Dd sol(rhs.size(), 0);
  mul(A, x, sol);
  sub(sol, rhs, sol);
  if (infnorm(sol, sol) > 1e-6)
    std::cerr << "WRONG PCG!\n";
  // std::cerr << sol << std::endl;
  hermes::ThreadArrayDistributionInfo td(pressure.resolution());
  __1To2<<<td.gridSize, td.blockSize>>>(A.indexDataAccessor(), x.accessor(),
                                        pressure.data().accessor());
}

__global__ void __projectionStepU(RegularGrid2Accessor<float> u, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (u.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    if (tex2D(solidTex2, xc - 1, yc))
      u(x, y) = 0; // tex2D(uSolidTex2, xc - 1, yc);
    else if (tex2D(solidTex2, xc, yc))
      u(x, y) = 0; // tex2D(uSolidTex2, xc, yc);
    else {
      float l = tex2D(pressureTex2, xc - 1, yc);
      float r = tex2D(pressureTex2, xc, yc);
      u(x, y) -= scale * (r - l);
    }
  }
}

__global__ void __projectionStepV(RegularGrid2Accessor<float> v, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (v.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    if (tex2D(solidTex2, xc, yc - 1))
      v(x, y) = 0; // tex2D(vSolidTex2, xc, yc - 1);
    else if (tex2D(solidTex2, xc, yc))
      v(x, y) = 0; // tex2D(vSolidTex2, xc, yc);
    else {
      float b = tex2D(pressureTex2, xc, yc - 1);
      float t = tex2D(pressureTex2, xc, yc);
      v(x, y) -= scale * (t - b);
    }
  }
}

template <>
void projectionStep_t(RegularGrid2Df &pressure, RegularGrid2Duc &solid,
                      StaggeredGrid2D &velocity, float dt) {
  pressureTex2.filterMode = cudaFilterModePoint;
  pressureTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
  Array2<float> pArray(pressure.resolution());
  Array2<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(pressureTex2, pArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, sArray.data(), channelDesc));
  memcpy(pArray, pressure.data());
  memcpy(sArray, solid.data());
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
  cudaUnbindTexture(pressureTex2);
  cudaUnbindTexture(solidTex2);
}

__global__ void __projectionStep(RegularGrid2Accessor<float> vel,
                                 RegularGrid2Accessor<float> pressure,
                                 RegularGrid2Accessor<unsigned char> solid,
                                 vec2u d, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (vel.isIndexStored(i, j)) {
    if (solid(i - d.x, j - d.y))
      vel(i, j) = 0; // tex3D(wSolidTex3, xc, yc, zc - 1);
    else if (solid(i, j))
      vel(i, j) = 0; // tex3D(wSolidTex3, xc, yc, zc);
    else {
      float b = pressure(i - d.x, j - d.y);
      float f = pressure(i, j);
      vel(i, j) -= scale * (f - b);
    }
  }
}

template <>
void projectionStep(RegularGrid2Df &pressure, RegularGrid2Duc &solid,
                    StaggeredGrid2D &velocity, float dt) {
  {
    auto info = velocity.u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), pressure.accessor(), solid.accessor(),
        vec2u(1, 0), scale);
  }
  {
    auto info = velocity.v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), pressure.accessor(), solid.accessor(),
        vec2u(0, 1), scale);
  }
}

} // namespace cuda

} // namespace poseidon
