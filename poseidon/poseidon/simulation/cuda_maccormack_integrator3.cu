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

texture<float, cudaTextureType3D> uTex3;
texture<float, cudaTextureType3D> vTex3;
texture<float, cudaTextureType3D> wTex3;
texture<float, cudaTextureType3D> phiTex3;
texture<float, cudaTextureType3D> phiNHatTex3;
texture<float, cudaTextureType3D> phiN1HatTex3;
texture<unsigned char, cudaTextureType3D> solidTex3;

__global__ void __computePhiN1_t(RegularGrid3Accessor<float> phi,
                                 RegularGrid3Info uInfo, RegularGrid3Info vInfo,
                                 RegularGrid3Info wInfo, float dt) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (phi.isIndexStored(x, y, z)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    unsigned char solid = tex3D(solidTex3, xc, yc, zc);
    if (solid) {
      phi(x, y, z) = 0;
      return;
    }
    point3f p = phi.worldPosition(x, y, z);
    point3f up = uInfo.toGrid(p) + vec3(0.5);
    point3f vp = vInfo.toGrid(p) + vec3(0.5);
    point3f wp = wInfo.toGrid(p) + vec3(0.5);
    vec3f vel(tex3D(uTex3, up.x, up.y, up.z), tex3D(vTex3, vp.x, vp.y, vp.z),
              tex3D(wTex3, wp.x, wp.y, wp.z));
    point3f npos = phi.gridPosition(p - vel * dt);
    point3f pos = floor(npos) + vec3(0.5);
    float nodeValues[8];
    nodeValues[0] = tex3D(phiTex3, pos.x, pos.y, pos.z);
    nodeValues[1] = tex3D(phiTex3, pos.x + 1, pos.y, pos.z);
    nodeValues[2] = tex3D(phiTex3, pos.x + 1, pos.y + 1, pos.z);
    nodeValues[3] = tex3D(phiTex3, pos.x, pos.y + 1, pos.z);
    nodeValues[4] = tex3D(phiTex3, pos.x, pos.y, pos.z + 1);
    nodeValues[5] = tex3D(phiTex3, pos.x + 1, pos.y, pos.z + 1);
    nodeValues[6] = tex3D(phiTex3, pos.x + 1, pos.y + 1, pos.z + 1);
    nodeValues[7] = tex3D(phiTex3, pos.x, pos.y + 1, pos.z + 1);
    float phiMin =
        min(nodeValues[7],
            min(nodeValues[6],
                min(nodeValues[5],
                    min(nodeValues[4],
                        min(nodeValues[3],
                            min(nodeValues[2],
                                min(nodeValues[0], nodeValues[1])))))));
    float phiMax =
        max(nodeValues[7],
            max(nodeValues[6],
                max(nodeValues[5],
                    max(nodeValues[4],
                        max(nodeValues[3],
                            max(nodeValues[2],
                                max(nodeValues[0], nodeValues[1])))))));

    phi(x, y, z) =
        tex3D(phiN1HatTex3, npos.x + 0.5, npos.y + 0.5, npos.z + 0.5) +
        0.5 * (tex3D(phiTex3, xc, yc, zc) - tex3D(phiNHatTex3, xc, yc, zc));
    phi(x, y, z) = max(min(phi(x, y, z), phiMax), phiMin);
  }
}

__global__ void __computePhiN1(StaggeredGrid3Accessor vel,
                               RegularGrid3Accessor<unsigned char> solid,
                               RegularGrid3Accessor<float> phiNHat,
                               RegularGrid3Accessor<float> phiN1Hat,
                               RegularGrid3Accessor<float> in,
                               RegularGrid3Accessor<float> out, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (in.isIndexStored(i, j, k)) {
    if (solid(i, j, k)) {
      out(i, j, k) = 0;
      return;
    }
    vec3f v = vel(i, j, k);
    point3f p = in.worldPosition(i, j, k);
    point3f wp = p - v * dt;
    point3f npos = in.gridPosition(wp);
    point3i pos(npos.x, npos.y, npos.z);
    float nodeValues[8];
    nodeValues[0] = in(pos.x, pos.y, pos.z);
    nodeValues[1] = in(pos.x + 1, pos.y, pos.z);
    nodeValues[2] = in(pos.x + 1, pos.y + 1, pos.z);
    nodeValues[3] = in(pos.x, pos.y + 1, pos.z);
    nodeValues[4] = in(pos.x, pos.y, pos.z + 1);
    nodeValues[5] = in(pos.x + 1, pos.y, pos.z + 1);
    nodeValues[6] = in(pos.x + 1, pos.y + 1, pos.z + 1);
    nodeValues[7] = in(pos.x, pos.y + 1, pos.z + 1);
    float phiMin =
        min(nodeValues[7],
            min(nodeValues[6],
                min(nodeValues[5],
                    min(nodeValues[4],
                        min(nodeValues[3],
                            min(nodeValues[2],
                                min(nodeValues[0], nodeValues[1])))))));
    float phiMax =
        max(nodeValues[7],
            max(nodeValues[6],
                max(nodeValues[5],
                    max(nodeValues[4],
                        max(nodeValues[3],
                            max(nodeValues[2],
                                max(nodeValues[0], nodeValues[1])))))));

    out(i, j, k) = phiN1Hat(wp) + 0.5 * (in(i, j, k) - phiNHat(i, j, k));
    out(i, j, k) = max(min(out(i, j, k), phiMax), phiMin);
  }
}

MacCormackIntegrator3::MacCormackIntegrator3() {
  uTex3.filterMode = cudaFilterModeLinear;
  uTex3.normalized = 0;
  vTex3.filterMode = cudaFilterModeLinear;
  vTex3.normalized = 0;
  wTex3.filterMode = cudaFilterModeLinear;
  wTex3.normalized = 0;
  phiTex3.filterMode = cudaFilterModeLinear;
  phiTex3.normalized = 0;
  phiNHatTex3.filterMode = cudaFilterModeLinear;
  phiNHatTex3.normalized = 0;
  phiN1HatTex3.filterMode = cudaFilterModeLinear;
  phiN1HatTex3.normalized = 0;
  solidTex3.filterMode = cudaFilterModePoint;
  solidTex3.normalized = 0;
}

void MacCormackIntegrator3::set(hermes::cuda::RegularGrid3Info info) {
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

void MacCormackIntegrator3::advect(StaggeredGrid3D &velocity,
                                   RegularGrid3Duc &solid, RegularGrid3Df &phi,
                                   RegularGrid3Df &phiOut, float dt) {

  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, solid, phi, phiN1Hat, dt);
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, solid, phiN1Hat, phiNHat, -dt);
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), phiNHat.accessor(),
      phiN1Hat.accessor(), phi.accessor(), phiOut.accessor(), dt);
}

void MacCormackIntegrator3::advect_t(VectorGrid3D &velocity,
                                     RegularGrid3Duc &solid,
                                     RegularGrid3Df &phi,
                                     RegularGrid3Df &phiOut, float dt) {

  // phi^_n+1 = A(phi_n)
  integrator.advect_t(velocity, solid, phi, phiN1Hat, dt);
  Array3<float> phiN1HatArray(phiN1Hat.resolution());
  memcpy(phiN1HatArray, phiN1Hat.data());
  CUDA_CHECK(cudaDeviceSynchronize());

  // phi^_n = Ar(phi^_n+1)
  integrator.advect_t(velocity, solid, phiN1Hat, phiNHat, -dt);
  Array3<float> phiNHatArray(phiNHat.resolution());
  memcpy(phiNHatArray, phiNHat.data());
  CUDA_CHECK(cudaDeviceSynchronize());

  Array3<unsigned char> solidArray(solid.resolution());
  Array3<float> uArray(velocity.u().resolution());
  Array3<float> vArray(velocity.v().resolution());
  Array3<float> wArray(velocity.w().resolution());
  Array3<float> phiArray(phi.resolution());
  memcpy(solidArray, solid.data());
  memcpy(uArray, velocity.u().data());
  memcpy(vArray, velocity.v().data());
  memcpy(wArray, velocity.w().data());
  memcpy(phiArray, phi.data());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaBindTextureToArray(solidTex3, solidArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaBindTextureToArray(uTex3, uArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(vTex3, vArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(wTex3, wArray.data(), channelDesc));
  CUDA_CHECK(cudaBindTextureToArray(phiTex3, phiArray.data(), channelDesc));
  CUDA_CHECK(
      cudaBindTextureToArray(phiNHatTex3, phiNHatArray.data(), channelDesc));
  CUDA_CHECK(
      cudaBindTextureToArray(phiN1HatTex3, phiN1HatArray.data(), channelDesc));

  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1_t<<<td.gridSize, td.blockSize>>>(
      phiOut.accessor(), velocity.u().info(), velocity.v().info(),
      velocity.w().info(), dt);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaUnbindTexture(solidTex3);
  cudaUnbindTexture(wTex3);
  cudaUnbindTexture(vTex3);
  cudaUnbindTexture(uTex3);
  cudaUnbindTexture(phiTex3);
  cudaUnbindTexture(phiNHatTex3);
  cudaUnbindTexture(phiN1HatTex3);
}

} // namespace cuda

} // namespace poseidon