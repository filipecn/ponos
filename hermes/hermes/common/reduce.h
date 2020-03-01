/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
/// \file reduce.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-02-25
///
/// \brief

#ifndef HERMES_REDUCE_H
#define HERMES_REDUCE_H

#include <hermes/common/cuda.h>
#include <hermes/common/defs.h>
#include <hermes/storage/array.h>
#include <hermes/storage/cuda_storage_utils.h>
#include <ponos/common/defs.h>

namespace hermes {

namespace cuda {

/// Set of pre-built predicates to use with reduction operations
struct ReducePredicates {
  /// \tparam T reduction input data type
  template <typename T> struct min {
    __host__ __device__ T operator()(const T &a) { return a; }
    __host__ __device__ T operator()(const T &a, const T &b) {
      return fminf(a, b);
    }
    __host__ __device__ T reduce(const T &a, const T &b) { return fminf(a, b); }
    T base_value = ponos::Constants::greatest<T>();
  };
  /// \tparam T reduction input data type
  template <typename T> struct min_abs {
    __host__ __device__ T operator()(const T &a) { return fabsf(a); }
    __host__ __device__ T operator()(const T &a, const T &b) {
      return fminf(fabsf(a), fabsf(b));
    }
    __host__ __device__ T reduce(const T &a, const T &b) { return fminf(a, b); }
    T base_value = ponos::Constants::greatest<T>();
  };
  /// \tparam T reduction input data type
  template <typename T> struct max {
    __host__ __device__ T operator()(const T &a) { return a; }
    __host__ __device__ T operator()(const T &a, const T &b) {
      return fmaxf(a, b);
    }
    __host__ __device__ T reduce(const T &a, const T &b) { return fmaxf(a, b); }
    T base_value = ponos::Constants::lowest<T>();
  };
  /// \tparam T reduction input data type
  template <typename T> struct max_abs {
    __host__ __device__ T operator()(const T &a) { return fabsf(a); }
    __host__ __device__ T operator()(const T &a, const T &b) {
      return fmaxf(fabsf(a), fabsf(b));
    }
    __host__ __device__ T reduce(const T &a, const T &b) { return fmaxf(a, b); }
    T base_value = ponos::Constants::lowest<T>();
  };
  /// \tparam T reduction input data type
  template <typename T> struct sum {
    __host__ __device__ T operator()(const T &a) { return a; }
    __host__ __device__ T operator()(const T &a, const T &b) { return a + b; }
    __host__ __device__ T reduce(const T &a, const T &b) { return a + b; }
    T base_value = 0;
  };
  /// \tparam T reduction input data type
  template <typename T> struct is_equal_to_value {
    __host__ __device__ is_equal_to_value(T value) : value(value) {}
    __host__ __device__ bool operator()(const T &a) {
      return Check::is_equal(a, value);
    }
    __host__ __device__ bool reduce(const bool &a, const bool &b) {
      return a && b;
    }
    T value{};
    bool base_value = true;
  };
  /// \tparam T reduction input data type
  template <typename T> struct is_equal {
    __host__ __device__ bool operator()(const T &a, const T &b) {
      return Check::is_equal(a, b);
    }
    __host__ __device__ bool reduce(const bool &a, const bool &b) {
      return a && b;
    }
    T value{};
    bool base_value = true;
  };
};

/// Auxiliary class that encapsulates a c++ lambda function into device code for
/// reduction operations in a single array
///\tparam T array data type
/// \tparam R reduction result data type
///\tparam F lambda function type following the signature: (index2, T&)
template <typename T, typename R, typename F> struct hermes_reduce_predicate {
  __host__ __device__ explicit hermes_reduce_predicate(const F &op)
      : predicate(op) {}
  __host__ __device__ R operator()(const T &a) { return predicate(a); }
  __host__ __device__ R reduce(const R &a, const R &b) {
    return predicate.reduce(a, b);
  }
  __host__ __device__ R baseValue() const { return predicate.base_value; }
  F predicate;
};
/// Auxiliary class that encapsulates a c++ lambda function into device code for
/// reduction operations in a pair of arrays
///\tparam T array data type
/// \tparam R reduction result data type
///\tparam F lambda function type following the signature: (index2, T&)
template <typename T, typename R, typename F> struct hermes_reduce2_predicate {
  __host__ __device__ explicit hermes_reduce2_predicate(const F &op)
      : predicate(op) {}
  __host__ __device__ R operator()(const T &a, const T &b) {
    return predicate(a, b);
  }
  __host__ __device__ R reduce(const R &a, const R &b) {
    return predicate.reduce(a, b);
  }
  __host__ __device__ R baseValue() const { return predicate.base_value; }
  F predicate;
};

/*****************************************************************************
************************     1-dimension single      *************************
******************************************************************************/

template <typename T, typename R, typename F>
__global__ void __reduce(Array1CAccessor<T> data, Array1Accessor<R> c,
                         hermes_reduce_predicate<T, R, F> predicate) {
  __shared__ R cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size();

  R temp = predicate.baseValue();
  while (tid < n) {
    temp = predicate.reduce(temp, predicate(data[tid]));
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] =
          predicate.reduce(cache[cacheindex], cache[cacheindex + i]);
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <typename T, typename R, typename ReducePredicate>
R reduce(const Array1<T> &data, ReducePredicate reduce_predicate) {
  size_t block_size = (data.size() + 256 - 1) / 256;
  if (block_size > 32)
    block_size = 32;
  hermes_reduce_predicate<T, R, ReducePredicate> hrp(reduce_predicate);
  Array1<R> d_c(block_size);
  __reduce<T, R, ReducePredicate>
      <<<block_size, 256>>>(data.constAccessor(), d_c.accessor(), hrp);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto h_c = d_c.hostData();
  R r = h_c[0];
  for (int i = 1; i < block_size; i++)
    r = reduce_predicate.reduce(r, h_c[i]);
  return r;
}

/*****************************************************************************
************************     1-dimension double      *************************
******************************************************************************/

template <typename T, typename R, typename F>
__global__ void __reduce(Array1CAccessor<T> data_a, Array1CAccessor<T> data_b,
                         Array1Accessor<R> c,
                         hermes_reduce2_predicate<T, R, F> predicate) {
  __shared__ R cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data_a.size();

  R temp = predicate.baseValue();
  while (tid < n) {
    temp = predicate.reduce(temp, predicate(data_a[tid], data_b[tid]));
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] =
          predicate.reduce(cache[cacheindex], cache[cacheindex + i]);
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <typename T, typename R, typename ReducePredicate>
R reduce(const Array1<T> &a, const Array1<T> &b,
         ReducePredicate reduce_predicate) {
  size_t block_size = (a.size() + 256 - 1) / 256;
  if (block_size > 32)
    block_size = 32;
  hermes_reduce2_predicate<T, R, ReducePredicate> hrp(reduce_predicate);
  Array1<R> d_c(block_size);
  __reduce<T, R, ReducePredicate><<<block_size, 256>>>(
      a.constAccessor(), b.constAccessor(), d_c.accessor(), hrp);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto h_c = d_c.hostData();
  R r = h_c[0];
  for (int i = 1; i < block_size; i++)
    r = reduce_predicate.reduce(r, h_c[i]);
  return r;
}

/*****************************************************************************
*************************        2-dimension         *************************
******************************************************************************/

template <typename T, typename R, typename F>
__global__ void __reduce(Array2CAccessor<T> data, Array1Accessor<R> c,
                         hermes_reduce_predicate<T, R, F> predicate) {
  __shared__ R cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size().total();

  R temp = predicate.baseValue();
  while (tid < n) {
    index2 ij(tid / data.size().width, tid % data.size().height);
    if (data.contains(ij))
      temp = predicate.reduce(temp, predicate(data[ij]));
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] =
          predicate.reduce(cache[cacheindex], cache[cacheindex + i]);
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <typename T, typename R, typename ReducePredicate>
R reduce(const Array2<T> &data, ReducePredicate reduce_predicate) {
  size_t block_size = (data.size().total() + 256 - 1) / 256;
  if (block_size > 32)
    block_size = 32;
  hermes_reduce_predicate<T, R, ReducePredicate> hrp(reduce_predicate);
  Array1<R> d_c(block_size);
  __reduce<T, R, ReducePredicate>
      <<<block_size, 256>>>(data.constAccessor(), d_c.accessor(), hrp);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto h_c = d_c.hostData();
  R r = h_c[0];
  for (int i = 1; i < block_size; i++) {
    r = reduce_predicate.reduce(r, h_c[i]);
  }
  return r;
}

/*
template <unsigned int blockSize, typename T>
__global__ void __reduceAdd(const T *data, T *rdata, unsigned int n) {
  extern __shared__ T sdata[];
  unsigned int threadId = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadId;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[threadId] = 0;
  while (i < n) {
    sdata[threadId] += data[i] + data[i + blockSize];
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (threadId < 256)
      sdata[threadId] += sdata[threadId + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadId < 128)
      sdata[threadId] += sdata[threadId + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadId < 64)
      sdata[threadId] += sdata[threadId + 64];
    __syncthreads();
  }
  if (threadId < 32) {
    if (blockSize >= 64)
      sdata[threadId] += sdata[threadId + 32];
    if (blockSize >= 32)
      sdata[threadId] += sdata[threadId + 16];
    if (blockSize >= 16)
      sdata[threadId] += sdata[threadId + 8];
    if (blockSize >= 8)
      sdata[threadId] += sdata[threadId + 4];
    if (blockSize >= 4)
      sdata[threadId] += sdata[threadId + 2];
    if (blockSize >= 2)
      sdata[threadId] += sdata[threadId + 1];
  }
  if (threadId == 0)
    rdata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, typename T>
__global__ void __reduceMin(const T *data, T *rdata, unsigned int n) {
  extern __shared__ T sdata[];
  unsigned int threadId = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadId;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[threadId] = Constants::lowest<T>();
  while (i < n) {
    sdata[threadId] = fminf(data[i], data[i + blockSize]);
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (threadId < 256)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 256]);
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadId < 128)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 128]);
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadId < 64)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 64]);
    __syncthreads();
  }
  if (threadId < 32) {
    if (blockSize >= 64)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 32]);
    if (blockSize >= 32)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 16]);
    if (blockSize >= 16)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 8]);
    if (blockSize >= 8)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 4]);
    if (blockSize >= 4)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 2]);
    if (blockSize >= 2)
      sdata[threadId] = fminf(sdata[threadId], sdata[threadId + 1]);
  }
  if (threadId == 0)
    rdata[blockIdx.x] = sdata[0];
}

template <typename T> T reduceAdd(const T *data, unsigned int n) {
  unsigned int blockSize = 128;
  unsigned int gridSize = 2;
  T h_r = 0;
  // __reduceAdd<blockSize, T><<<gridSize, blockSize>>>(data, r, n);
  return h_r;
}*/

template <typename T> __global__ void k__min(Array2Accessor<T> data, T *c) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size().width * data.size().height;

  T temp = 1 << 20;
  while (tid < n) {
    index2 ij(tid / data.size().width, tid % data.size().height);
    printf("%d, ", data[ij]);
    temp = fminf(temp, data[ij]);
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] = fminf(cache[cacheindex], cache[cacheindex + i]);
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <typename T> T minValue(Array2<T> &data) {
  size_t blockSize = (data.size().width * data.size().height + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  k__min<<<blockSize, 256>>>(data.accessor(), d_c);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T norm = 0;
  for (int i = 0; i < blockSize; i++) {
    PING;
    std::cerr << c[i] << std::endl;
    norm = fmin(norm, c[i]);
  }
  cudaFree(d_c);
  delete[] c;
  return norm;
}

template <typename T> __global__ void k__max(Array2Accessor<T> data, T *c) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size().x * data.size().y;

  T temp = -1 << 20;
  while (tid < n) {
    int i = tid / data.size().x;
    int j = tid % data.size().x;
    temp = fmaxf(temp, data(i, j));
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

template <typename T> T maxValue(Array2<T> &data) {
  size_t blockSize = (data.size().x * data.size().y + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  k__max<<<blockSize, 256>>>(data.accessor(), d_c);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T norm = 0;
  for (int i = 0; i < blockSize; i++)
    norm = fmax(norm, c[i]);
  cudaFree(d_c);
  delete[] c;
  return norm;
}

template <typename T> __global__ void __max_abs(Array2Accessor<T> data, T *c) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size().x * data.size().y;

  T temp = -1 << 20;
  while (tid < n) {
    int i = tid / data.size().x;
    int j = tid % data.size().x;
    temp = fmaxf(temp, fabsf(data(i, j)));
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

template <typename T> T maxAbs(Array2<T> &data) {
  size_t blockSize = (data.size().x * data.size().y + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  __max_abs<<<blockSize, 256>>>(data.accessor(), d_c);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T norm = 0;
  for (int i = 0; i < blockSize; i++)
    norm = fmax(norm, c[i]);
  cudaFree(d_c);
  delete[] c;
  return norm;
}

template <typename T>
__global__ void __max_abs(MemoryBlock3Accessor<T> data, T *c) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.size().x * data.size().y * data.size().z;

  T temp = -1 << 20;
  while (tid < n) {
    int k = tid / (data.size().x * data.size().y);
    int j = (tid % (data.size().x * data.size().y)) / data.size().x;
    int i = tid % data.size().x;
    temp = fmaxf(temp, fabsf(data(i, j, k)));
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

template <typename T> T maxAbs(MemoryBlock3<MemoryLocation::DEVICE, T> &data) {
  size_t blockSize =
      (data.size().x * data.size().y * data.size().z + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  T *c = new T[blockSize];
  T *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(T));
  __max_abs<<<blockSize, 256>>>(data.accessor(), d_c);
  cudaMemcpy(c, d_c, blockSize * sizeof(T), cudaMemcpyDeviceToHost);
  T norm = 0;
  for (int i = 0; i < blockSize; i++)
    norm = fmax(norm, c[i]);
  cudaFree(d_c);
  delete[] c;
  return norm;
}

} // namespace cuda

} // namespace hermes

#endif