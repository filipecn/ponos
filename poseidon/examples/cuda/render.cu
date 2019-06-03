#include "render.h"

using namespace hermes::cuda;

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ int rgbToInt(float r, float g, float b, float a) {
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  a = clamp(a, 0.0f, 255.0f);
  return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void __renderScalarGradient(RegularGrid2Accessor<float> in,
                                       unsigned int *out, hermes::cuda::Color a,
                                       hermes::cuda::Color b, float minValue,
                                       float maxValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(x, y)) {
    auto v = fabs(in(x, y));
    auto t = (v - minValue) / (maxValue - minValue);
    auto c = hermes::cuda::mixsRGB(a, b, t);
    c = hermes::cuda::toRGB(c);
    out[y * in.resolution().x + x] = rgbToInt(c.r, c.g, c.b, 255);
  }
}

void renderDistances(RegularGrid2Df &in, unsigned int *out, Color a, Color b) {
  using namespace hermes::cuda;
  std::cerr << "distances range:\t" << min(in) << " " << max(in) << " -> "
            << maxAbs(in.data()) << std::endl;
  auto td = hermes::ThreadArrayDistributionInfo(in.resolution());
  __renderScalarGradient<<<td.gridSize, td.blockSize>>>(in.accessor(), out, a,
                                                        b, 0.f, 1.f);
}