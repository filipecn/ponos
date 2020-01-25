#include "cuda_texture_kernels.h"
#include <hermes/common/cuda.h>
#include <hermes/storage/cuda_storage_utils.h>

namespace hermes {

namespace cuda {

surface<void, 3> tex3Out;

__global__ void __surfWrite(cudaPitchedPtr data, cudaExtent size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= size.width || y >= size.height || z >= size.depth)
    return;

  char *devPtr = (char *)data.ptr;
  size_t pitch = data.pitch;
  size_t slicePitch = pitch * size.height;
  char *slice = devPtr + z * slicePitch;
  float *row = (float *)(slice + y * pitch);
  float output = row[x];
  // surface writes need byte offsets for x!
  surf3Dwrite(output, tex3Out, x * sizeof(float), y, z);
}

template <typename T>
__global__ void __fillTexture(T *data, T value, unsigned int w,
                              unsigned int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h)
    data[y * w + x] = value;
}

template <typename T>
__global__ void __fillTexture(cudaPitchedPtr data, T value, unsigned int w,
                              unsigned int h, unsigned int d) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < w && y < h && z < d) {
    // char *devPtr = (char *)data.ptr;
    // size_t pitch = data.pitch;
    // size_t slicePitch = pitch * h;
    // char *slice = devPtr + z * slicePitch;
    // float *row = (float *)(slice + y * pitch);
    // row[x] = value;
    // row[x] = z * 100 + y * 10 + x;
    // T *ptr = (T *)((char *)data.ptr +
    //                pitchedIndexOffset<T>(data.pitch, w, h, x, y, z));
    // *ptr = z * 100 + y * 10 + x;
    // *pitchedIndexPtr<T>(data, x, y, z) = z * 100 + y * 10 + x;
    pitchedIndexRef<T>(data, x, y, z) = value;
  }
}

void fillTexture(float *data, float value, unsigned int w, unsigned int h) {
  ThreadArrayDistributionInfo td(w, h);
  __fillTexture<float><<<td.gridSize, td.blockSize>>>(data, value, w, h);
}

void fillTexture(cudaPitchedPtr data, float value, unsigned int w,
                 unsigned int h, unsigned int d) {
  ThreadArrayDistributionInfo td(w, h, d);
  __fillTexture<float><<<td.gridSize, td.blockSize>>>(data, value, w, h, d);
}

void copyToArray3D(cudaPitchedPtr data, cudaArray *array, cudaExtent size) {
  ThreadArrayDistributionInfo td(size.width, size.height, size.depth);
  CHECK_CUDA(cudaBindSurfaceToArray(tex3Out, array));
  __surfWrite<<<td.gridSize, td.blockSize>>>(data, size);
}

} // namespace cuda

} // namespace hermes