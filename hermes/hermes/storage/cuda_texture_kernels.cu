#include "cuda_texture_kernels.h"

namespace hermes {

namespace cuda {

template <typename T>
__global__ void __fillTexture(T *data, T value, unsigned int w,
                              unsigned int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h)
    data[y * w + x] = value;
}

void fillTexture(float *data, float value, unsigned int w, unsigned int h) {
  ThreadArrayDistributionInfo td(w, h);
  __fillTexture<float><<<td.gridSize, td.blockSize>>>(data, value, w, h);
}

} // namespace cuda

} // namespace hermes