#include "utils.h"

__global__ void __checkBoard(float *output, unsigned int w, unsigned int h) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    output[y * w + x] = 1.f;
  }
}

void fillTexture(hermes::cuda::Texture<float> &t) {
  hermes::ThreadArrayDistributionInfo td(t.width(), t.height());
  __checkBoard<<<td.gridSize, td.blockSize>>>(t.deviceData(), t.width(),
                                              t.height());
}
