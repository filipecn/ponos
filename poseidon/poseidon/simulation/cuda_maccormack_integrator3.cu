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

texture<float, cudaTextureType3D> uTex3;
texture<float, cudaTextureType3D> vTex3;
texture<float, cudaTextureType3D> wTex3;
texture<float, cudaTextureType3D> phiTex3;
texture<float, cudaTextureType3D> phiNHatTex3;
texture<float, cudaTextureType3D> phiN1HatTex3;
texture<unsigned char, cudaTextureType3D> solidTex3;

__global__ void __computePhiN1(cudaPitchedPtr phi,
                               hermes::cuda::Grid3Info phiInfo,
                               hermes::cuda::Grid3Info uInfo,
                               hermes::cuda::Grid3Info vInfo,
                               hermes::cuda::Grid3Info wInfo, float dt) {
  using namespace hermes::cuda;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < phiInfo.resolution.x && y < phiInfo.resolution.y) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    float zc = z + 0.5;
    unsigned char solid = tex3D(solidTex3, xc, yc, zc);
    if (solid) {
      pitchedIndexRef<float>(phi, x, y, z) = 0;
      return;
    }
    point3f p = phiInfo.toWorld(point3f(x, y, z));
    point3f up = uInfo.toField(p) + vec3(0.5);
    point3f vp = vInfo.toField(p) + vec3(0.5);
    point3f wp = wInfo.toField(p) + vec3(0.5);
    vec3f vel(tex3D(uTex3, up.x, up.y, up.z), tex3D(vTex3, vp.x, vp.y, vp.z),
              tex3D(wTex3, wp.x, wp.y, wp.z));
    point3f npos = phiInfo.toField(p - vel * dt);
    point3f pos = floor(npos) + vec3(0.5);
    float nodeValues[4];
    nodeValues[0] = tex3D(phiTex3, pos.x, pos.y, pos.z);
    nodeValues[1] = tex3D(phiTex3, pos.x + 1, pos.y, pos.z);
    nodeValues[2] = tex3D(phiTex3, pos.x + 1, pos.y + 1, pos.z);
    nodeValues[3] = tex3D(phiTex3, pos.x, pos.y + 1, pos.z);
    float phiMin = min(nodeValues[3],
                       min(nodeValues[2], min(nodeValues[0], nodeValues[1])));
    float phiMax = max(nodeValues[3],
                       max(nodeValues[2], max(nodeValues[0], nodeValues[1])));
    pitchedIndexRef<float>(phi, x, y, z) = 0;
    tex3D(phiN1HatTex3, npos.x + 0.5, npos.y + 0.5, npos.z + 0.5) +
        0.5 * (tex3D(phiTex3, xc, yc, zc) - tex3D(phiNHatTex3, xc, yc, zc));
    pitchedIndexRef<float>(phi, x, y, z) =
        max(min(pitchedIndexRef<float>(phi, x, y, z), phiMax), phiMin);
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

void MacCormackIntegrator3::advect(
    const hermes::cuda::VectorGrid3D &velocity,
    const hermes::cuda::RegularGrid3<hermes::cuda::MemoryLocation::DEVICE,
                                     unsigned char> &solid,
    const hermes::cuda::RegularGrid3Df &phi,
    hermes::cuda::RegularGrid3Df &phiOut, float dt) {
  using namespace hermes::cuda;
  // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  // CUDA_CHECK(cudaBindTextureToArray(solidTex3,
  // solid.texture().textureArray(),
  //                                   channelDesc));
  // channelDesc = cudaCreateChannelDesc<float>();
  // CUDA_CHECK(cudaBindTextureToArray(
  //     uTex3, velocity.u().texture().textureArray(), channelDesc));
  // CUDA_CHECK(cudaBindTextureToArray(
  //     vTex3, velocity.v().texture().textureArray(), channelDesc));
  // CUDA_CHECK(cudaBindTextureToArray(
  //     wTex3, velocity.w().texture().textureArray(), channelDesc));
  // CUDA_CHECK(cudaBindTextureToArray(
  //     phiNHatTex3, phiNHat.texture().textureArray(), channelDesc));
  // CUDA_CHECK(cudaBindTextureToArray(
  //     phiN1HatTex3, phiN1Hat.texture().textureArray(), channelDesc));
  // CUDA_CHECK(cudaBindTextureToArray(phiTex3, phi.texture().textureArray(),
  //                                   channelDesc));
  // auto info = phi.info();
  // hermes::ThreadArrayDistributionInfo td(info.resolution);
  // // phi^_n+1 = A(phi_n)
  // integrator.advect(velocity, solid, phi, phiN1Hat, dt);
  // CUDA_CHECK(cudaDeviceSynchronize());
  // phiN1Hat.texture().updateTextureMemory();
  // CUDA_CHECK(cudaDeviceSynchronize());
  // // phi^_n = Ar(phi^_n+1)
  // integrator.advect(velocity, solid, phiN1Hat, phiNHat, -dt);
  // CUDA_CHECK(cudaDeviceSynchronize());
  // phiNHat.texture().updateTextureMemory();
  // CUDA_CHECK(cudaDeviceSynchronize());
  // // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  // __computePhiN1<<<td.gridSize, td.blockSize>>>(
  //     phiOut.texture().pitchedData(), phi.info(), velocity.u().info(),
  //     velocity.v().info(), velocity.w().info(), dt);
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