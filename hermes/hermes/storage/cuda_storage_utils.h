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

#ifndef HERMES_STORAGE_CUDA_STORAGE_UTILS_H
#define HERMES_STORAGE_CUDA_STORAGE_UTILS_H

#include <hermes/common/cuda.h>

namespace hermes {
namespace cuda {

template <typename T>
__host__ __device__ T &pitchedIndexRef(cudaPitchedPtr data, size_t i, size_t j,
                                       size_t k) {
  size_t w = data.xsize / sizeof(T);
  return (T &)(*((char *)data.ptr + k * (data.pitch * data.ysize) +
                 j * data.pitch + i * sizeof(T)));
}

template <typename T>
__host__ __device__ T *pitchedIndexPtr(cudaPitchedPtr data, size_t i, size_t j,
                                       size_t k) {
  size_t w = data.xsize / sizeof(T);
  return (T *)((char *)data.ptr + k * (data.pitch * data.ysize) +
               j * data.pitch + i * sizeof(T));
}

template <typename T>
__host__ __device__ size_t pitchedIndexOffset(size_t pitch, size_t w, size_t h,
                                              size_t i, size_t j, size_t k) {
  return k * (pitch * h) + j * pitch + i * sizeof(T);
}

template <typename T>
void copyPitchedToLinear(cudaPitchedPtr pitchedMemory, T *linearMemory,
                         cudaMemcpyKind kind, unsigned int depth) {
  cudaMemcpy3DParms p = {0};
  int width = pitchedMemory.xsize / sizeof(T);
  p.srcPtr.ptr = pitchedMemory.ptr;
  p.srcPtr.pitch = pitchedMemory.pitch;
  p.srcPtr.xsize = width;
  p.srcPtr.ysize = pitchedMemory.ysize;
  p.dstPtr.ptr = linearMemory;
  p.dstPtr.pitch = width * sizeof(T);
  p.dstPtr.xsize = width;
  p.dstPtr.ysize = pitchedMemory.ysize;
  p.extent.width = width * sizeof(T);
  p.extent.height = pitchedMemory.ysize;
  p.extent.depth = depth;
  p.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&p));
}

template <typename T>
void copyLinearToPitched(T *linearMemory, cudaPitchedPtr pitchedMemory,
                         cudaMemcpyKind kind, unsigned int depth) {
  cudaMemcpy3DParms p = {0};
  int width = pitchedMemory.xsize / sizeof(T);
  p.dstPtr.ptr = pitchedMemory.ptr;
  p.dstPtr.pitch = pitchedMemory.pitch;
  p.dstPtr.xsize = width;
  p.dstPtr.ysize = pitchedMemory.ysize;
  p.srcPtr.ptr = linearMemory;
  p.srcPtr.pitch = width * sizeof(T);
  p.srcPtr.xsize = width;
  p.srcPtr.ysize = pitchedMemory.ysize;
  p.extent.width = width * sizeof(T);
  p.extent.height = pitchedMemory.ysize;
  p.extent.depth = depth;
  p.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&p));
}

} // namespace cuda
} // namespace hermes

#endif