#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

// Simple transformation kernel
// __global__ void __transform(float *output, TextureMemoryInfo info,
//                             float theta) {
//   // Calculate normalized texture coordinates
//   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

//   float u = x / (float)info.width;
//   float v = y / (float)info.height;

//   // Transform coordinates
//   u -= 0.5f;
//   v -= 0.5f;
//   float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
//   float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

//   // Read from texture and write to global memory
//   output[y * info.width + x] = tex2D<float>(info.obj, tu, tv);
// }

TEST(Texture, fill) {
  unsigned int width = 30;
  unsigned int height = 30;
  Texture<float> t(width, height);
  fillTexture(t);
  // fillTexture<float>(nullptr, 2, 2, 2);
  std::vector<float> h_data(width * height);
  CUDA_CHECK(cudaMemcpy(h_data.data(), t.deviceData(),
                        h_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
  print(h_data.data(), width, height);
}