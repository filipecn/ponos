/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
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

#ifndef HERMES_COMMON_CUDA_H
#define HERMES_COMMON_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace hermes {

namespace cuda {

static void handleCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    std::exit(-1);
  }
}

#define CHECK_CUDA(err) (handleCudaError(err, __FILE__, __LINE__))

inline void print_cuda_devices() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Major compute capability: %d\n", prop.major);
    printf("  Minor compute capability: %d\n", prop.minor);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Device can map host memory with "
           "cudaHostAlloc/cudaHostGetDevicePointer: %d\n",
           prop.canMapHostMemory);
    printf("  Clock frequency in kilohertz: %d\n", prop.clockRate);
    printf("  Compute mode (See cudaComputeMode): %d\n", prop.computeMode);
    printf("  Device can concurrently copy memory and execute a kernel: %d\n",
           prop.deviceOverlap);
    printf("  Device is integrated as opposed to discrete: %d\n",
           prop.integrated);
    printf("  Specified whether there is a run time limit on kernels: %d\n",
           prop.kernelExecTimeoutEnabled);
    printf("  Maximum size of each dimension of a grid: %d %d %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Maximum size of each dimension of a block: %d %d %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Maximum pitch in bytes allowed by memory copies: %u\n",
           prop.memPitch);
    printf("  Number of multiprocessors on device: %d\n",
           prop.multiProcessorCount);
    printf("  32-bit registers available per block: %d\n", prop.regsPerBlock);
    printf("  Shared memory available per block in bytes: %d\n",
           prop.sharedMemPerBlock);
    printf("  Alignment requirement for textures: %d\n", prop.textureAlignment);
    printf("  Constant memory available on device in bytes: %u\n",
           prop.totalConstMem);
    printf("  Global memory available on device in bytes: %u\n",
           prop.totalGlobalMem);
    printf("  Warp size in threads: %d\n", prop.warpSize);
  }
}

inline void print_cuda_memory_usage() {
  size_t free_byte;
  size_t total_byte;
  CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}

#define CUDA_MEMORY_USAGE                                                      \
  {                                                                            \
    std::cerr << "[INFO][" << __FILE__ << "][" << __LINE__ << "]";             \
    print_cuda_memory_usage();                                                 \
  }

} // namespace cuda

} // namespace hermes

#endif // HERMES_COMMON_CUDA_H