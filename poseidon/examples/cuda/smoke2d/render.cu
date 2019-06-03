#include "render.h"
#include <cuda_runtime.h>
#include <hermes/hermes.h>

texture<float, cudaTextureType2D> scalarTex2;
texture<float, cudaTextureType2D> densityTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;

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
    auto value = tex2D(densityTex2, x / float(w), y / float(h)) * 255;
    uchar4 c4 = make_uchar4(value, value, value, 255);
    out[y * w + x] = rgbToInt(c4.x, c4.y, c4.z, c4.w);
  }
}

__global__ void __renderSolids(unsigned int *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    unsigned char solid = tex2D(solidTex2, x / float(w), y / float(h)) * 255;
    if (solid > 0) {
      uchar4 c4 = make_uchar4(solid, 100, 0, 255);
      out[y * w + x] = rgbToInt(c4.x, c4.y, c4.z, c4.w);
    }
  }
}

__global__ void __renderScalarGradient(unsigned int *out, int w, int h,
                                       hermes::cuda::Color a,
                                       hermes::cuda::Color b, float minValue,
                                       float maxValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    auto v = tex2D(scalarTex2, x / float(w), y / float(h));
    auto t = (v - minValue) / (maxValue - minValue);
    auto c = hermes::cuda::mixsRGB(a, b, t);
    c = hermes::cuda::toRGB(c);
    out[y * w + x] = rgbToInt(c.r, c.g, c.b, 255);
  }
}

void renderScalarGradient(unsigned int w, unsigned int h,
                          const hermes::cuda::Texture<float> &in,
                          unsigned int *out, float minValue, float maxValue,
                          hermes::cuda::Color a, hermes::cuda::Color b) {
  auto td = hermes::ThreadArrayDistributionInfo(w, h);
  scalarTex2.normalized = 1;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  using namespace hermes::cuda;
  CUDA_CHECK(
      cudaBindTextureToArray(scalarTex2, in.textureArray(), channelDesc));
  __renderScalarGradient<<<td.gridSize, td.blockSize>>>(out, w, h, a, b,
                                                        minValue, maxValue);
  cudaUnbindTexture(scalarTex2);
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

void renderDensity(hermes::cuda::RegularGrid2Df &in, unsigned int *out) {
  hermes::cuda::Array2<float> pArray(in.resolution());
  hermes::cuda::memcpy(pArray, in.data());
  auto td = hermes::ThreadArrayDistributionInfo(in.resolution());
  densityTex2.normalized = 1;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  using namespace hermes::cuda;
  CUDA_CHECK(cudaBindTextureToArray(densityTex2, pArray.data(), channelDesc));
  __renderDensity<<<td.gridSize, td.blockSize>>>(out, in.resolution().x,
                                                 in.resolution().y);
  cudaUnbindTexture(densityTex2);
}

void renderSolids(unsigned int w, unsigned int h,
                  const hermes::cuda::Texture<unsigned char> &in,
                  unsigned int *out) {
  auto td = hermes::ThreadArrayDistributionInfo(w, h);
  solidTex2.normalized = 1;
  solidTex2.filterMode = cudaFilterModePoint;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  using namespace hermes::cuda;
  CUDA_CHECK(cudaBindTextureToArray(solidTex2, in.textureArray(), channelDesc));
  __renderSolids<<<td.gridSize, td.blockSize>>>(out, w, h);
  cudaUnbindTexture(solidTex2);
}