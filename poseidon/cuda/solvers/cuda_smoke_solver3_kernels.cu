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

texture<float, cudaTextureType3D> uTex3, uCopyTex3, wuTex3;
texture<float, cudaTextureType3D> vTex3, vCopyTex3, wvTex3;
texture<float, cudaTextureType3D> wTex3, wCopyTex3, wwTex3;
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
    s(x, y, z) += dt * source(x, y, z) * 0.5;
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

__global__ void __applyForceField_t(RegularGrid3Accessor<float> velocity,
                                    RegularGrid3Accessor<float> force,
                                    float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (velocity.isIndexStored(x, y, z)) {
    auto wp = velocity.worldPosition(x, y, z);
    auto p = force.gridPosition(wp) + vec3f(0.5f);
    // printf("%f ", tex3D(forceTex3, p.x, p.y, p.z));
    velocity(x, y, z) += dt * tex3D(forceTex3, p.x, p.y, p.z);
  }
}

template <>
void applyForceField_t(StaggeredGrid3D &velocity, VectorGrid3D &forceField,
                       float dt) {
  forceTex3.filterMode = cudaFilterModeLinear;
  forceTex3.normalized = 0;
  Array3<float> forceArray(forceField.u().resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(forceTex3, forceArray.data(), channelDesc));
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
  {
    memcpy(forceArray, forceField.w().data());
    hermes::ThreadArrayDistributionInfo td(velocity.w().resolution());
    __applyForceField_t<<<td.gridSize, td.blockSize>>>(
        velocity.w().accessor(), forceField.w().accessor(), dt);
  }
  cudaUnbindTexture(forceTex3);
}

__global__ void __applyForceField(RegularGrid3Accessor<float> velocity,
                                  RegularGrid3Accessor<unsigned char> solid,
                                  RegularGrid3Accessor<float> force, float dt,
                                  vec3u d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (velocity.isIndexStored(i, j, k) && !solid(i, j, k) &&
      !solid(i - d.x, j - d.y, k - d.z)) {
    velocity(i, j, k) +=
        dt * (force(i - d.x, j - d.y, k - d.z) + force(i, j, k)) * 0.5f;
  }
}

template <>
void applyForceField(StaggeredGrid3D &velocity, RegularGrid3Duc &solid,
                     VectorGrid3D &forceField, float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), solid.accessor(), forceField.u().accessor(),
        dt, vec3u(1, 0, 0));
  }
  {
    hermes::ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), solid.accessor(), forceField.v().accessor(),
        dt, vec3u(0, 1, 0));
  }
  {
    hermes::ThreadArrayDistributionInfo td(velocity.w().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.w().accessor(), solid.accessor(), forceField.w().accessor(),
        dt, vec3u(0, 0, 1));
  }
}

__global__ void __applyBuoyancyForceField_t(
    RegularGrid3Accessor<float> fu, RegularGrid3Accessor<float> fv,
    RegularGrid3Accessor<float> fw, RegularGrid3Accessor<float> density,
    RegularGrid3Accessor<float> temperature, float tamb, float alpha,
    float beta) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (fu.isIndexStored(x, y, z)) {
    auto wp = fu.worldPosition(x, y, z);
    auto p = temperature.gridPosition(wp) + vec3f(0.5f);
    // printf("%f %f %f (%f %f %f)= %f %f\n", wp.x, wp.y, wp.z, p.x, p.y, p.z,
    //        density(x, y, z), temperature(x, y, z));
    fu(x, y, z) = fw(x, y, z) = 0.f;
    fv(x, y, z) += -alpha * tex3D(densityTex3, p.x, p.y, p.z) +
                   beta * (tex3D(temperatureTex3, p.x, p.y, p.z) - tamb);
  }
}

template <>
void computeBuoyancyForceField_t(VectorGrid3D &forceField,
                                 RegularGrid3Df &density,
                                 RegularGrid3Df &temperature,
                                 float ambientTemperature, float alpha,
                                 float beta) {
  temperatureTex3.filterMode = cudaFilterModeLinear;
  temperatureTex3.normalized = 0;
  densityTex3.filterMode = cudaFilterModeLinear;
  densityTex3.normalized = 0;
  Array3<float> tArray(temperature.resolution());
  Array3<float> dArray(density.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(
      cudaBindTextureToArray(temperatureTex3, tArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(densityTex3, dArray.data(), channelDesc));
  memcpy(tArray, temperature.data());
  memcpy(dArray, density.data());
  hermes::ThreadArrayDistributionInfo td(forceField.resolution());
  __applyBuoyancyForceField_t<<<td.gridSize, td.blockSize>>>(
      forceField.u().accessor(), forceField.v().accessor(),
      forceField.w().accessor(), density.accessor(), temperature.accessor(),
      ambientTemperature, alpha, beta);
  cudaUnbindTexture(temperatureTex3);
  cudaUnbindTexture(densityTex3);
}

__global__ void __addBuoyancyForce(VectorGrid3Accessor f,
                                   RegularGrid3Accessor<unsigned char> solid,
                                   RegularGrid3Accessor<float> density,
                                   RegularGrid3Accessor<float> temperature,
                                   float tamb, float alpha, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (f.vAccessor().isIndexStored(i, j, k) && !solid(i, j, k))
    f.v(i, j, k) += 9.81 * (-alpha * density(i, j, k) +
                            beta * (temperature(i, j, k) - tamb));
}

template <>
void addBuoyancyForce(VectorGrid3D &forceField, RegularGrid3Duc &solid,
                      RegularGrid3Df &density, RegularGrid3Df &temperature,
                      float ambientTemperature, float alpha, float beta) {
  hermes::ThreadArrayDistributionInfo td(forceField.resolution());
  __addBuoyancyForce<<<td.gridSize, td.blockSize>>>(
      forceField.accessor(), solid.accessor(), density.accessor(),
      temperature.accessor(), ambientTemperature, alpha, beta);
}

__device__ float u_ijk(float i, float j, float k) {
  return (tex3D(uTex3, i + 1, j, k) + tex3D(uTex3, i, j, k)) / 2.f;
}
__device__ float v_ijk(float i, float j, float k) {
  return (tex3D(vTex3, i, j + 1, k) + tex3D(vTex3, i, j, k)) / 2.f;
}
__device__ float w_ijk(float i, float j, float k) {
  return (tex3D(wTex3, i, j, k + 1) + tex3D(wTex3, i, j, k)) / 2.f;
}

__global__ void __computeVorticity_t(RegularGrid3Accessor<float> wu,
                                     RegularGrid3Accessor<float> wv,
                                     RegularGrid3Accessor<float> ww) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (wu.isIndexStored(x, y, z)) {
    float i = x + 0.5;
    float j = y + 0.5;
    float k = z + 0.5;
    float inv = 1.f / (2.f * wu.spacing().x);
    wu(x, y, z) = (w_ijk(i, j + 1, k) - w_ijk(i, j - 1, k) -
                   v_ijk(i, j, k + 1) + v_ijk(i, j, k - 1)) *
                  inv;
    inv = 1.f / (2.f * wu.spacing().y);
    wv(x, y, z) = (u_ijk(i, j, k + 1) - u_ijk(i, j, k - 1) -
                   w_ijk(i + 1, j, k) + w_ijk(i - 1, j, k)) *
                  inv;
    inv = 1.f / (2.f * wu.spacing().z);
    ww(x, y, z) = (v_ijk(i + 1, j, k) - v_ijk(i - 1, j, k) -
                   u_ijk(i, j + 1, k) + u_ijk(i, j - 1, k)) *
                  inv;
  }
}

template <>
void computeVorticity_t(StaggeredGrid3D &velocity, RegularGrid3Duc &solid,
                        VectorGrid3D &vorticityField) {
  uTex3.filterMode = cudaFilterModePoint;
  uTex3.normalized = 0;
  vTex3.filterMode = cudaFilterModePoint;
  vTex3.normalized = 0;
  wTex3.filterMode = cudaFilterModePoint;
  wTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
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
  hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
  __computeVorticity_t<<<td.gridSize, td.blockSize>>>(
      vorticityField.u().accessor(), vorticityField.v().accessor(),
      vorticityField.w().accessor());
  cudaUnbindTexture(uTex3);
  cudaUnbindTexture(vTex3);
  cudaUnbindTexture(wTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __computeVorticityNorm_t(RegularGrid3Accessor<float> n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (n.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    if (tex3D(solidTex3, xc, yc, zc)) {
      n(x, y, z) = 0.f;
      return;
    }
    vec3f w(tex3D(wuTex3, xc, yc, zc), tex3D(wvTex3, xc, yc, zc),
            tex3D(wwTex3, xc, yc, zc));
    // n(x, y, z) = w.length();
  }
}

void computeVorticityNorm_t(RegularGrid3Duc &solid,
                            VectorGrid3D &vorticityField,
                            RegularGrid3Df &wNorm) {
  wuTex3.filterMode = cudaFilterModePoint;
  wuTex3.normalized = 0;
  wvTex3.filterMode = cudaFilterModePoint;
  wvTex3.normalized = 0;
  wwTex3.filterMode = cudaFilterModePoint;
  wwTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
  Array3<float> wuArray(vorticityField.u().resolution());
  Array3<float> wvArray(vorticityField.v().resolution());
  Array3<float> wwArray(vorticityField.w().resolution());
  Array3<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(wuTex3, wuArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wvTex3, wvArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wwTex3, wwArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, sArray.data(), channelDesc));
  memcpy(wuArray, vorticityField.u().data());
  memcpy(wvArray, vorticityField.v().data());
  memcpy(wwArray, vorticityField.w().data());
  memcpy(sArray, solid.data());
  hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
  __computeVorticityNorm_t<<<td.gridSize, td.blockSize>>>(wNorm.accessor());
  cudaUnbindTexture(wuTex3);
  cudaUnbindTexture(wvTex3);
  cudaUnbindTexture(wwTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __computeVorticityComfinementForce_t(
    RegularGrid3Accessor<float> fu, RegularGrid3Accessor<float> fv,
    RegularGrid3Accessor<float> fw, RegularGrid3Accessor<float> wNorm,
    float eta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (wNorm.isIndexStored(i, j, k)) {
    float xc = i + 0.5;
    float yc = j + 0.5;
    float zc = k + 0.5;
    if (tex3D(solidTex3, xc, yc, zc)) {
      fu(i, j, k) = 0.f;
      fv(i, j, k) = 0.f;
      fw(i, j, k) = 0.f;
      return;
    }
    // wNorm gradient
    vec3f inv(1.f / (2 * wNorm.spacing().x), 1.f / (2 * wNorm.spacing().y),
              1.f / (2 * wNorm.spacing().z));
    vec3f g((wNorm(i + 1, j, k) - wNorm(i - 1, j, k)) * inv.x,
            (wNorm(i, j + 1, k) - wNorm(i, j - 1, k)) * inv.y,
            (wNorm(i, j, k + 1) - wNorm(i, j, k - 1)) * inv.z);
    // normalized
    vec3f N = g / (float)(g.length() + 1e-20 * wNorm.spacing().x);
    vec3f w(tex3D(wuTex3, xc, yc, zc), tex3D(wvTex3, xc, yc, zc),
            tex3D(wwTex3, xc, yc, zc));
    vec3f force = cross(N, w);
    fu(i, j, k) = eta * wNorm.spacing().x * force.x;
    fu(i, j, k) = eta * wNorm.spacing().y * force.y;
    fu(i, j, k) = eta * wNorm.spacing().z * force.z;
  }
}

template <>
void computeVorticityConfinementForceField_t(StaggeredGrid3D &velocity,
                                             RegularGrid3Duc &solid,
                                             VectorGrid3D &vorticityField,
                                             VectorGrid3D &forceField,
                                             float eta, float dt) {
  computeVorticity_t(velocity, solid, vorticityField);
  // std::cerr << "VORTICITY U\n";
  // std::cerr << vorticityField.u().data() << std::endl;
  // std::cerr << "VORTICITY V\n";
  // std::cerr << vorticityField.v().data() << std::endl;
  // std::cerr << "VORTICITY W\n";
  // std::cerr << vorticityField.w().data() << std::endl;
  RegularGrid3Df wNorm(solid.resolution());
  computeVorticityNorm_t(solid, vorticityField, wNorm);
  // compute force
  RegularGrid3Df confForce(solid.resolution());
  wuTex3.filterMode = cudaFilterModePoint;
  wuTex3.normalized = 0;
  wvTex3.filterMode = cudaFilterModePoint;
  wvTex3.normalized = 0;
  wwTex3.filterMode = cudaFilterModePoint;
  wwTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
  Array3<float> wuArray(vorticityField.u().resolution());
  Array3<float> wvArray(vorticityField.v().resolution());
  Array3<float> wwArray(vorticityField.w().resolution());
  Array3<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(wuTex3, wuArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wvTex3, wvArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wwTex3, wwArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, sArray.data(), channelDesc));
  memcpy(wuArray, vorticityField.u().data());
  memcpy(wvArray, vorticityField.v().data());
  memcpy(wwArray, vorticityField.w().data());
  memcpy(sArray, solid.data());
  hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
  __computeVorticityComfinementForce_t<<<td.gridSize, td.blockSize>>>(
      forceField.u().accessor(), forceField.v().accessor(),
      forceField.w().accessor(), wNorm.accessor(), eta);
  cudaUnbindTexture(wuTex3);
  cudaUnbindTexture(wvTex3);
  cudaUnbindTexture(wwTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __computeVorticity(VectorGrid3Accessor vor,
                                   StaggeredGrid3Accessor vel) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (vor.uAccessor().isIndexStored(i, j, k)) {
    float inv = 1.f / (2.f * vel.uAccessor().spacing().x);
    vor.u(i, j, k) = (vel(i, j + 1, k).z - vel(i, j - 1, k).z -
                      vel(i, j, k + 1).y + vel(i, j, k - 1).y) *
                     inv;
    inv = 1.f / (2.f * vel.vAccessor().spacing().y);
    vor.v(i, j, k) = (vel(i, j, k + 1).x - vel(i, j, k - 1).x -
                      vel(i + 1, j, k).z + vel(i - 1, j, k).z) *
                     inv;
    inv = 1.f / (2.f * vel.wAccessor().spacing().z);
    vor.w(i, j, k) = (vel(i + 1, j, k).y - vel(i - 1, j, k).y -
                      vel(i, j + 1, k).x + vel(i, j - 1, k).x) *
                     inv;
  }
}

__global__ void
__computeVorticityNorm(RegularGrid3Accessor<float> n,
                       RegularGrid3Accessor<unsigned char> solid,
                       VectorGrid3Accessor vor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (n.isIndexStored(i, j, k)) {
    if (solid(i, j, k)) {
      n(i, j, k) = 0.f;
      return;
    }
    vec3f w = vor(i, j, k);
    n(i, j, k) = w.length();
  }
}

__global__ void __computeVorticityComfinementForce(
    VectorGrid3Accessor f, VectorGrid3Accessor vor,
    RegularGrid3Accessor<unsigned char> solid,
    RegularGrid3Accessor<float> wNorm, float eta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (wNorm.isIndexStored(i, j, k)) {
    if (solid(i, j, k)) {
      // f.u(i, j, k) = 0.f;
      // f.v(i, j, k) = 0.f;
      // f.w(i, j, k) = 0.f;
      return;
    }
    // wNorm gradient
    vec3f inv(1.f / (2 * wNorm.spacing().x), 1.f / (2 * wNorm.spacing().y),
              1.f / (2 * wNorm.spacing().z));
    vec3f g((wNorm(i + 1, j, k) - wNorm(i - 1, j, k)) * inv.x,
            (wNorm(i, j + 1, k) - wNorm(i, j - 1, k)) * inv.y,
            (wNorm(i, j, k + 1) - wNorm(i, j, k - 1)) * inv.z);
    // normalized
    vec3f N =
        g / (float)(g.length() + 1e-20 * (1.f / (wNorm.spacing().x * 0.01)));
    vec3f w = vor(i, j, k);
    vec3f force = cross(N, w);
    f.u(i, j, k) += eta * wNorm.spacing().x * force.x;
    f.v(i, j, k) += eta * wNorm.spacing().y * force.y;
    f.w(i, j, k) += eta * wNorm.spacing().z * force.z;
  }
}

template <>
void addVorticityConfinementForce(VectorGrid3D &forceField,
                                  StaggeredGrid3D &velocity,
                                  RegularGrid3Duc &solid,
                                  VectorGrid3D &vorticityField, float eta,
                                  float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
    __computeVorticity<<<td.gridSize, td.blockSize>>>(vorticityField.accessor(),
                                                      velocity.accessor());
  }
  RegularGrid3Df wNorm(solid.resolution());
  wNorm.setSpacing(vorticityField.spacing());
  {
    hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
    __computeVorticityNorm<<<td.gridSize, td.blockSize>>>(
        wNorm.accessor(), solid.accessor(), vorticityField.accessor());
  }
  // compute force
  RegularGrid3Df confForce(solid.resolution());
  hermes::ThreadArrayDistributionInfo td(vorticityField.resolution());
  __computeVorticityComfinementForce<<<td.gridSize, td.blockSize>>>(
      forceField.accessor(), vorticityField.accessor(), solid.accessor(),
      wNorm.accessor(), eta);
}

__global__ void __computeDivergence_t(RegularGrid3Accessor<float> divergence,
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
void computeDivergence_t(
    StaggeredGrid3D &velocity,
    RegularGrid3<MemoryLocation::DEVICE, unsigned char> &solid,
    RegularGrid3Df &divergence) {
  uTex3.filterMode = cudaFilterModePoint;
  uTex3.normalized = 0;
  vTex3.filterMode = cudaFilterModePoint;
  vTex3.normalized = 0;
  wTex3.filterMode = cudaFilterModePoint;
  wTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
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
  vec3f inv(-1.f / divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence_t<<<td.gridSize, td.blockSize>>>(divergence.accessor(),
                                                       inv);
  cudaUnbindTexture(uTex3);
  cudaUnbindTexture(vTex3);
  cudaUnbindTexture(wTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __computeDivergence(StaggeredGrid3Accessor vel,
                                    RegularGrid3Accessor<unsigned char> solid,
                                    RegularGrid3Accessor<float> divergence,
                                    vec3f invdx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (divergence.isIndexStored(i, j, k)) {
    float left = vel.u(i, j, k);
    float right = vel.u(i + 1, j, k);
    float bottom = vel.v(i, j, k);
    float top = vel.v(i, j + 1, k);
    float back = vel.w(i, j, k);
    float front = vel.w(i, j, k + 1);
    unsigned char sleft = solid(i - 1, j, k);
    unsigned char sright = solid(i + 1, j, k);
    unsigned char sbottom = solid(i, j - 1, k);
    unsigned char stop = solid(i, j + 1, k);
    unsigned char sback = solid(i, j, k - 1);
    unsigned char sfront = solid(i, j, k + 1);
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
    divergence(i, j, k) =
        dot(invdx, vec3f(right - left, top - bottom, front - back));
  }
}

template <>
void computeDivergence(StaggeredGrid3D &velocity, RegularGrid3Duc &solid,
                       RegularGrid3Df &divergence) {
  auto info = divergence.info();
  vec3f inv(-1.f / divergence.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), divergence.accessor(), inv);
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
    } else
      A(i, j, k).diag += scale;
    // bottom - top
    if (solid.isIndexStored(i, j - 1, k) && !solid(i, j - 1, k))
      A(i, j, k).diag += scale;
    if (solid.isIndexStored(i, j + 1, k)) {
      if (!solid(i, j + 1, k)) {
        A(i, j, k).diag += scale;
        A(i, j, k).y = -scale;
      } // else // EMPTY
      //   A(i, j, k).diag += scale;
    } else
      A(i, j, k).diag += scale;
    // back - front
    if (solid.isIndexStored(i, j, k - 1) && !solid(i, j, k - 1))
      A(i, j, k).diag += scale;
    if (solid.isIndexStored(i, j, k + 1)) {
      if (!solid(i, j, k + 1)) {
        A(i, j, k).diag += scale;
        A(i, j, k).z = -scale;
      } //  else // EMPTY
        //   A(i, j, k).diag += scale;
    } else
      A(i, j, k).diag += scale;
  }
}

__global__ void __buildRHS(MemoryBlock3Accessor<int> indices,
                           RegularGrid3Accessor<float> divergence,
                           MemoryBlock1Accessor<double> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (indices.isIndexValid(i, j, k)) {
    if (indices(i, j, k) >= 0)
      rhs[indices(i, j, k)] = divergence(i, j, k);
  }
}

__global__ void __1To3(MemoryBlock3Accessor<int> indices,
                       MemoryBlock1Accessor<double> v,
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
                           MemoryBlock1Dd &rhs) {
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
void solvePressureSystem(FDMatrix3D &A, RegularGrid3Df &divergence,
                         RegularGrid3Df &pressure, RegularGrid3Duc &solid,
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
  pcg(x, A, rhs, rhs.size(), 1e-8);
  // std::cerr << residual << "\n" << x << std::endl;
  // store pressure values
  MemoryBlock1Dd sol(rhs.size(), 0);
  mul(A, x, sol);
  sub(sol, rhs, sol);
  if (infnorm(sol, sol) > 1e-6)
    std::cerr << "WRONG PCG!\n";
  // std::cerr << sol << std::endl;
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
void projectionStep_t(RegularGrid3Df &pressure, RegularGrid3Duc &solid,
                      StaggeredGrid3D &velocity, float dt) {
  pressureTex3.filterMode = cudaFilterModePoint;
  pressureTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
  Array3<float> pArray(pressure.resolution());
  Array3<unsigned char> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(pressureTex3, pArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, sArray.data(), channelDesc));
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
  {
    auto info = velocity.w().info();
    float invdx = 1.0 / info.spacing.z;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepW<<<td.gridSize, td.blockSize>>>(velocity.w().accessor(),
                                                     scale);
  }
  cudaUnbindTexture(pressureTex3);
  cudaUnbindTexture(solidTex3);
}

__global__ void __projectionStep(RegularGrid3Accessor<float> vel,
                                 RegularGrid3Accessor<float> pressure,
                                 RegularGrid3Accessor<unsigned char> solid,
                                 vec3u d, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (vel.isIndexStored(i, j, k)) {
    if (solid(i - d.x, j - d.y, k - d.z))
      vel(i, j, k) = 0; // tex3D(wSolidTex3, xc, yc, zc - 1);
    else if (solid(i, j, k))
      vel(i, j, k) = 0; // tex3D(wSolidTex3, xc, yc, zc);
    else {
      float b = pressure(i - d.x, j - d.y, k - d.z);
      float f = pressure(i, j, k);
      vel(i, j, k) -= scale * (f - b);
    }
  }
}

template <>
void projectionStep(RegularGrid3Df &pressure, RegularGrid3Duc &solid,
                    StaggeredGrid3D &velocity, float dt) {
  {
    auto info = velocity.u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), pressure.accessor(), solid.accessor(),
        vec3u(1, 0, 0), scale);
  }
  {
    auto info = velocity.v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), pressure.accessor(), solid.accessor(),
        vec3u(0, 1, 0), scale);
  }
  {
    auto info = velocity.w().info();
    float invdx = 1.0 / info.spacing.z;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.w().accessor(), pressure.accessor(), solid.accessor(),
        vec3u(0, 0, 1), scale);
  }
}

} // namespace cuda

} // namespace poseidon
