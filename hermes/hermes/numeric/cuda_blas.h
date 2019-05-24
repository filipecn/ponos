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

#ifndef HERMES_NUMERIC_CUDA_BLAS_H
#define HERMES_NUMERIC_CUDA_BLAS_H

#include <hermes/common/cuda_parallel.h>
#include <hermes/storage/cuda_storage_utils.h>

namespace hermes {

namespace cuda {

template <typename T> __global__ void __axpy(T a, T *x, T *y, T *r, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    r[i] = a * x[i] + y[i];
}

template <typename T> __global__ void __sub(T *a, T *b, T *c, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] - b[i];
}

template <typename T>
__global__ void __dot(const T *a, const T *b, T *c, size_t n) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;

  T temp = 0;
  while (tid < n) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] += cache[cacheindex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <typename T> __global__ void __infNorm(const T *a, T *c, size_t n) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;

  T temp = 0;
  while (tid < n) {
    temp = fmaxf(temp, fabsf(a[tid]));
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] = fmaxf(cache[cacheindex], cache[cacheindex + i]);
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

// Calculates r = a*x + y
template <typename T> void axpy(T a, T *x, T *y, T *r, size_t n) {
  ThreadArrayDistributionInfo td(n);
  __axpy<<<td.gridSize, td.blockSize>>>(a, x, y, r, n);
}

template <typename T> T dot(const T *a, const T *b, size_t n) {
  size_t blockSize = (n + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  __dot<<<blockSize, 256>>>(a, b, d_c, n);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T sum = 0;
  for (int i = 0; i < blockSize; i++)
    sum += c[i];
  cudaFree(d_c);
  delete[] c;
  return sum;
}
// Calculate infinity norm max |a[i]|
template <typename T> T infnorm(const T *a, size_t n) {
  size_t blockSize = (n + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  __infNorm<<<blockSize, 256>>>(a, d_c, n);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T norm = 0;
  for (int i = 0; i < blockSize; i++)
    norm = fmax(norm, c[i]);
  cudaFree(d_c);
  delete[] c;
  return norm;
}
// c = a - b
template <typename T> void sub(T *a, T *b, T *c, size_t n) {
  ThreadArrayDistributionInfo td(n);
  __sub<<<td.gridSize, td.blockSize>>>(a, b, c, n);
}

// r = dot(a,b)
template <typename T>
T dot(MemoryBlock1<MemoryLocation::DEVICE, T> &a,
      MemoryBlock1<MemoryLocation::DEVICE, T> &b,
      MemoryBlock1<MemoryLocation::DEVICE, T> &w) {
  return dot(a.ptr(), b.ptr(), a.size());
}
// r = max |a[i]|
template <typename T>
T infnorm(MemoryBlock1<MemoryLocation::DEVICE, T> &a,
          MemoryBlock1<MemoryLocation::DEVICE, T> &w) {
  return infnorm(a.ptr(), a.size());
}
// r = a * x + y
template <typename T>
void axpy(T a, MemoryBlock1<MemoryLocation::DEVICE, T> &x,
          MemoryBlock1<MemoryLocation::DEVICE, T> &y,
          MemoryBlock1<MemoryLocation::DEVICE, T> &r) {
  axpy(a, x.ptr(), y.ptr(), r.ptr(), x.size());
}
// r = a - b
template <typename T>
void sub(MemoryBlock1<MemoryLocation::DEVICE, T> &a,
         MemoryBlock1<MemoryLocation::DEVICE, T> &b,
         MemoryBlock1<MemoryLocation::DEVICE, T> &r) {
  sub(a.ptr(), b.ptr(), r.ptr(), a.size());
}

} // namespace cuda

} // namespace hermes

#endif