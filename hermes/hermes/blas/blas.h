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

#include <hermes/blas/vector.h>
#include <hermes/common/cuda_parallel.h>
#include <hermes/storage/cuda_memory_block.h>
#include <hermes/storage/cuda_storage_utils.h>

namespace hermes {

namespace cuda {

/// Set of methods that compose the BLAS
class BLAS {
public:
  /// Dot product
  /// \tparam T
  /// \param a **[in]**
  /// \param b **[in]**
  /// \return T
  template <typename T> static T dot(const Vector<T> &a, const Vector<T> &b) {
    return reduce<T, T, dot_predicate<T>>(a.data(), b.data(),
                                          dot_predicate<T>());
  }
  /// Peforms r = a * x + y
  /// \tparam T
  /// \param a **[in]**
  /// \param x **[in]**
  /// \param y **[in]**
  /// \param r **[out]**
  template <typename T>
  static void axpy(T a, const Vector<T> &x, const Vector<T> &y, Vector<T> &r) {
    // r = a * x + y;
    compute(x.data().constAccessor(), y.data().constAccessor(),
            r.data().accessor(), axpy_operation<T>(a));
  }
  /// Computes max |v[i]|
  /// \tparam T
  /// \param v **[in]**
  /// \return T
  template <typename T> static T infNorm(const Vector<T> &v) {
    return reduce<T, T, ReducePredicates::max_abs<T>>(
        v.data(), ReducePredicates::max_abs<T>());
  }
  ///
  /// \tparam T
  template <typename T> struct dot_predicate {
    __host__ __device__ T operator()(const T &a, const T &b) { return a * b; }
    __host__ __device__ T reduce(const T &a, const T &b) { return a + b; }
    T base_value{0};
  };
  template <typename T> struct axpy_operation {
    __host__ __device__ axpy_operation(T a) : a(a) {}
    __host__ __device__ void operator()(const T &x, const T &y, T &r) {
      r = a * x + y;
    }
    T a{0};
  };
};

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
T dot(CuMemoryBlock1<T> &a, CuMemoryBlock1<T> &b, CuMemoryBlock1<T> &w) {
  return dot(a.ptr(), b.ptr(), a.size());
}
// r = max |a[i]|
template <typename T> T infnorm(CuMemoryBlock1<T> &a, CuMemoryBlock1<T> &w) {
  return infnorm(a.ptr(), a.size());
}
// r = a * x + y
template <typename T>
void axpy(T a, CuMemoryBlock1<T> &x, CuMemoryBlock1<T> &y,
          CuMemoryBlock1<T> &r) {
  axpy(a, x.ptr(), y.ptr(), r.ptr(), x.size());
}
// r = a - b
template <typename T>
void sub(CuMemoryBlock1<T> &a, CuMemoryBlock1<T> &b, CuMemoryBlock1<T> &r) {
  sub(a.ptr(), b.ptr(), r.ptr(), a.size());
}

} // namespace cuda

} // namespace hermes

#endif