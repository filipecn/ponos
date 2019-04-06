#include "render.h"
#include <cuda_runtime.h>
#include <hermes/hermes.h>

texture<float, cudaTextureType2D> densityTex2;

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ int rgbToInt(float r, float g, float b, float a) {
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  a = clamp(a, 0.0f, 255.0f);
  return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void __renderDensity(unsigned int *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    uchar4 c4 = make_uchar4(
        tex2D(densityTex2, x / float(w), y / float(h)) * 100, 0, 0, 255);
    out[y * w + x] = rgbToInt(c4.x, c4.y, c4.z, c4.w);
  }
}

void renderDensity(unsigned int w, unsigned int h,
                   const hermes::cuda::Texture<float> &in, unsigned int *out) {
  auto td = hermes::ThreadArrayDistributionInfo(w, h);
  // densityTex2.addressMode[0] = cudaAddressModeBorder;
  // densityTex2.addressMode[1] = cudaAddressModeBorder;
  // densityTex2.filterMode = cudaFilterModeLinear;
  densityTex2.normalized = 1;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  using namespace hermes::cuda;
  CUDA_CHECK(
      cudaBindTextureToArray(densityTex2, in.textureArray(), channelDesc));
  __renderDensity<<<td.gridSize, td.blockSize>>>(out, w, h);
  cudaUnbindTexture(densityTex2);
}