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
#include <hermes/common/cuda_parallel.h>
#include <hermes/common/defs.h>
#include <hermes/geometry/cuda_vector.h>
#include <hermes/storage/cuda_array.h>
#include <iomanip>
#include <vector>

namespace hermes {
namespace cuda {

inline cudaMemcpyKind copyDirection(MemoryLocation src, MemoryLocation dst) {
  if (src == MemoryLocation::DEVICE && dst == MemoryLocation::DEVICE)
    return cudaMemcpyDeviceToDevice;
  if (src == MemoryLocation::DEVICE && dst == MemoryLocation::HOST)
    return cudaMemcpyDeviceToHost;
  if (src == MemoryLocation::HOST && dst == MemoryLocation::HOST)
    return cudaMemcpyHostToHost;
  return cudaMemcpyHostToDevice;
}

template <typename T>
__host__ __device__ T &pitchedIndexRef(cudaPitchedPtr data, size_t i, size_t j,
                                       size_t k) {
  return (T &)(*((char *)data.ptr + k * (data.pitch * data.ysize) +
                 j * data.pitch + i * sizeof(T)));
}

template <typename T>
__host__ __device__ T *pitchedIndexPtr(cudaPitchedPtr data, size_t i, size_t j,
                                       size_t k) {
  return (T *)((char *)data.ptr + k * (data.pitch * data.ysize) +
               j * data.pitch + i * sizeof(T));
}

template <typename T>
__host__ __device__ size_t pitchedIndexOffset(size_t pitch, size_t w, size_t h,
                                              size_t i, size_t j, size_t k) {
  return k * (pitch * h) + j * pitch + i * sizeof(T);
}

template <typename T>
void copyLinearToLinear(T *dst, const T *src, cudaMemcpyKind kind, vec3u size) {
  CUDA_CHECK(cudaMemcpy(dst, src, size.x * size.y * size.z * sizeof(T), kind));
}

template <typename T>
void copyPitchedToPitched(cudaPitchedPtr dst, cudaPitchedPtr src,
                          cudaMemcpyKind kind) {
  CUDA_CHECK(cudaMemcpy2D(dst.ptr, dst.pitch, src.ptr, src.pitch, src.xsize,
                          src.ysize, kind));
}

template <typename T>
void copyPitchedToPitched(cudaPitchedPtr dst, cudaPitchedPtr src,
                          cudaMemcpyKind kind, unsigned int depth) {
  cudaMemcpy3DParms p = {0};
  int width = src.xsize / sizeof(T);
  p.srcPtr.ptr = src.ptr;
  p.srcPtr.pitch = src.pitch;
  p.srcPtr.xsize = width;
  p.srcPtr.ysize = src.ysize;
  p.dstPtr.ptr = dst.ptr;
  p.dstPtr.pitch = dst.pitch;
  p.dstPtr.xsize = width;
  p.dstPtr.ysize = dst.ysize;
  p.extent.width = width * sizeof(T);
  p.extent.height = dst.ysize;
  p.extent.depth = depth;
  p.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&p));
}

template <typename T>
void copyPitchedToLinear(T *linearMemory, cudaPitchedPtr pitchedMemory,
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
void copyLinearToPitched(cudaPitchedPtr pitchedMemory, T *linearMemory,
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

template <typename T>
void copyPitchedToArray(cudaArray *dst, cudaPitchedPtr src,
                        cudaMemcpyKind kind) {
  CUDA_CHECK(cudaMemcpy2DToArray(dst, 0, 0, src.ptr, src.pitch, src.xsize,
                                 src.ysize, kind));
}

template <typename T>
void copyPitchedToArray(cudaArray *dst, cudaPitchedPtr src, cudaMemcpyKind kind,
                        size_t depth) {
  cudaMemcpy3DParms copyParams = {0};
  int width = src.xsize / sizeof(T);
  cudaExtent extent = make_cudaExtent(width, src.ysize, depth);
  copyParams.srcPtr = src;
  copyParams.dstArray = dst;
  copyParams.extent = extent;
  copyParams.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

template <typename T> class MemoryBlock1Accessor {
public:
  MemoryBlock1Accessor(T *data, size_t size) : size_(size), data_(data) {}
  __host__ __device__ size_t size() const { return size_; }
  __host__ __device__ T &operator[](size_t i) { return data_[i]; }
  __host__ __device__ const T &operator[](size_t i) const { return data_[i]; }
  __host__ __device__ bool isIndexValid(int i) const {
    return i >= 0 && i < (int)size_;
  }

protected:
  size_t size_;
  T *data_ = nullptr;
};

template <MemoryLocation L, typename T> class MemoryBlock1 {};

template <typename T>
__global__ void __fill1(MemoryBlock1Accessor<T> data, T value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  auto size = data.size();
  if (i < size)
    data[i] = value;
}

template <typename T> void fill1(MemoryBlock1Accessor<T> data, T value) {
  ThreadArrayDistributionInfo td(data.size());
  __fill1<T><<<td.gridSize, td.blockSize>>>(data, value);
}

template <MemoryLocation L, typename T>
void fill1(MemoryBlock1<L, T> data, T value) {
  ThreadArrayDistributionInfo td(data.size());
  __fill1<T><<<td.gridSize, td.blockSize>>>(data.accessor(), value);
}

template <typename T> class MemoryBlock1<MemoryLocation::DEVICE, T> {
public:
  MemoryBlock1(size_t size = 0) : size_(size) {}
  MemoryBlock1(size_t size, const T &value) : size_(size) {
    allocate();
    ThreadArrayDistributionInfo td(size);
    __fill1<T><<<td.gridSize, td.blockSize>>>(this->accessor(), value);
  }
  void resize(size_t size) { size_ = size; }
  ~MemoryBlock1() {
    if (data_)
      cudaFree(data_);
  }
  MemoryLocation location() const { return MemoryLocation::DEVICE; }
  size_t size() const { return size_; }
  void allocate() {
    if (data_)
      cudaFree(data_);
    CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
  }
  size_t memorySize() { return size_ * sizeof(T); }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock1Accessor<T> accessor() {
    return MemoryBlock1Accessor<T>(data_, size_);
  }

private:
  size_t size_;
  T *data_ = nullptr;
};

template <typename T> class MemoryBlock1<MemoryLocation::HOST, T> {
public:
  MemoryBlock1(size_t size = 0) : size_(size) {}
  MemoryBlock1(size_t size, const T &value) : size_(size) {
    allocate();
    // TODO: init
  }
  void resize(size_t size) { size_ = size; }
  ~MemoryBlock1() {
    if (data_)
      delete[] data_;
  }
  MemoryLocation location() const { return MemoryLocation::HOST; }
  size_t size() const { return size_; }
  void allocate() {
    if (data_)
      delete[] data_;
    data_ = new T[size_];
  }
  size_t memorySize() { return size_ * sizeof(T); }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock1Accessor<T> accessor() {
    return MemoryBlock1Accessor<T>(data_, size_);
  }

private:
  size_t size_;
  T *data_ = nullptr;
};

template <typename T> class MemoryBlock2Iterator {
public:
  class Element {
  public:
    __host__ __device__ Element(T &v, const vec2i &ij) : value(v), index_(ij) {}
    __host__ __device__ vec2i index() const { return index_; }
    __host__ __device__ int i() const { return index_.x; }
    __host__ __device__ int j() const { return index_.y; }

    T &value;

  private:
    vec2i index_;
  };
  __host__ __device__ MemoryBlock2Iterator(T *data, const vec2u &size,
                                           size_t pitch, const vec2i &ij)
      : size_(size), data_(data), pitch_(pitch), i(ij.x), j(ij.y) {}
  __host__ __device__ vec2u size() const { return size_; }
  __host__ __device__ MemoryBlock2Iterator &operator++() {
    i++;
    if (i >= size_.x) {
      i = 0;
      j++;
      if (j >= size_.y) {
        i = j = -1;
      }
    }
    return *this;
  }
  __host__ __device__ Element operator*() {
    return Element((T &)(*((char *)data_ + j * pitch_ + i * sizeof(T))),
                   vec2i(i, j));
  }
  __host__ __device__ bool operator==(const MemoryBlock2Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i &&
           j == other.j && pitch_ == other.pitch_;
  }
  __host__ __device__ bool operator!=(const MemoryBlock2Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i ||
           j != other.j || pitch_ != other.pitch_;
  }

private:
  int i = 0, j = 0;
  vec2u size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
};

template <typename T> class MemoryBlock2Accessor {
public:
  MemoryBlock2Accessor(T *data, const vec2u &size, size_t pitch)
      : size_(size), data_(data), pitch_(pitch) {}
  __host__ __device__ vec2u size() const { return size_; }
  __host__ __device__ T &operator()(size_t i, size_t j) {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  __host__ __device__ const T &operator()(size_t i, size_t j) const {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  __host__ __device__ bool isIndexValid(int i, int j) const {
    return i >= 0 && i < (int)size_.x && j >= 0 && j < (int)size_.y;
  }
  __host__ __device__ MemoryBlock2Iterator<T> begin() {
    return MemoryBlock2Iterator<T>(data_, size_, pitch_, vec2i(0));
  }
  __host__ __device__ MemoryBlock2Iterator<T> end() {
    return MemoryBlock2Iterator<T>(data_, size_, pitch_, vec2i(-1));
  }

protected:
  vec2u size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
};

template <MemoryLocation L, typename T> class MemoryBlock2 {};

template <typename T> class MemoryBlock2<MemoryLocation::DEVICE, T> {
public:
  MemoryBlock2(const vec2u &size = vec2u()) : size_(size) {}
  void resize(const vec2u &size) { size_ = size; }
  ~MemoryBlock2() {
    if (data_)
      cudaFree(data_);
  }
  MemoryLocation location() const { return MemoryLocation::DEVICE; }
  const vec2u &size() const { return size_; }
  void allocate() {
    if (data_)
      cudaFree(data_);
    void *pdata = nullptr;
    CUDA_CHECK(cudaMallocPitch(&pdata, &pitch_, size_.x * sizeof(T), size_.y));
    data_ = reinterpret_cast<T *>(pdata);
  }
  size_t memorySize() { return pitch_ * size_.x * size_.y * sizeof(T); }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock2Accessor<T> accessor() {
    return MemoryBlock2Accessor<T>(data_, size_, pitch_);
  }
  cudaPitchedPtr pitchedData() {
    cudaPitchedPtr pd{0};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.x * sizeof(T);
    pd.ysize = size_.y;
    return pd;
  }

private:
  vec2u size_;
  size_t pitch_ = 0;
  T *data_ = nullptr;
};

template <typename T> class MemoryBlock2<MemoryLocation::HOST, T> {
public:
  MemoryBlock2(const vec2u &size = vec2u()) : size_(size) {}
  void resize(const vec2u &size) { size_ = size; }
  ~MemoryBlock2() {
    if (data_)
      delete[] data_;
  }
  MemoryLocation location() const { return MemoryLocation::HOST; }
  const vec2u &size() const { return size_; }
  void allocate() {
    if (data_)
      delete[] data_;
    pitch_ = size_.x * sizeof(T);
    data_ = new T[size_.x * size_.y];
  }
  size_t memorySize() { return size_.x * size_.y * sizeof(T); }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock2Accessor<T> accessor() {
    return MemoryBlock2Accessor<T>(data_, size_, pitch_);
  }
  cudaPitchedPtr pitchedData() {
    cudaPitchedPtr pd{0};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.x * sizeof(T);
    pd.ysize = size_.y;
    return pd;
  }

private:
  vec2u size_;
  size_t pitch_ = 0;
  T *data_ = nullptr;
};

template <typename T> class MemoryBlock3Iterator {
public:
  class Element {
  public:
    __host__ __device__ Element(T &v, const vec3i &ijk)
        : value(v), index_(ijk) {}
    __host__ __device__ vec3i index() const { return index_; }
    __host__ __device__ int i() const { return index_.x; }
    __host__ __device__ int j() const { return index_.y; }
    __host__ __device__ int k() const { return index_.z; }

    T &value;

  private:
    vec3i index_;
  };
  __host__ __device__ MemoryBlock3Iterator(T *data, const vec3u &size,
                                           size_t pitch, const vec3i &ijk)
      : size_(size), data_(data), pitch_(pitch), i(ijk.x), j(ijk.y), k(ijk.z) {}
  __host__ __device__ vec3u size() const { return size_; }
  __host__ __device__ MemoryBlock3Iterator &operator++() {
    i++;
    if (i >= size_.x) {
      i = 0;
      j++;
      if (j >= size_.y) {
        i = 0;
        j = 0;
        k++;
        if (k >= size_.z)
          i = j = k = -1;
      }
    }
    return *this;
  }
  __host__ __device__ Element operator*() {
    return Element((T &)(*((char *)data_ + k * (pitch_ * size_.y) + j * pitch_ +
                           i * sizeof(T))),
                   vec3i(i, j, k));
  }
  __host__ __device__ bool operator==(const MemoryBlock3Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i &&
           j == other.j && k == other.k && pitch_ == other.pitch_;
  }
  __host__ __device__ bool operator!=(const MemoryBlock3Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i ||
           j != other.j || k != other.k || pitch_ != other.pitch_;
  }

private:
  int i = 0, j = 0, k = 0;
  vec3u size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
};

template <typename T> class MemoryBlock3Accessor {
public:
  __host__ __device__ MemoryBlock3Accessor(const MemoryBlock3Accessor &other) {
    size_ = other.size_;
    data_ = other.data_;
    pitch_ = other.pitch_;
  }
  MemoryBlock3Accessor(T *data, const vec3u &size, size_t pitch)
      : size_(size), data_(data), pitch_(pitch) {}
  __host__ __device__ vec3u size() const { return size_; }
  __host__ __device__ T &operator()(size_t i, size_t j, size_t k) {
    return (T &)(*((char *)data_ + k * (pitch_ * size_.y) + j * pitch_ +
                   i * sizeof(T)));
  }
  __host__ __device__ const T &operator()(size_t i, size_t j, size_t k) const {
    return (T &)(*((char *)data_ + k * (pitch_ * size_.y) + j * pitch_ +
                   i * sizeof(T)));
  }
  __host__ __device__ bool isIndexValid(int i, int j, int k) const {
    return i >= 0 && i < (int)size_.x && j >= 0 && j < (int)size_.y && k >= 0 &&
           k < (int)size_.z;
  }
  __host__ __device__ MemoryBlock3Iterator<T> begin() {
    return MemoryBlock3Iterator<T>(data_, size_, pitch_, vec3i(0));
  }
  __host__ __device__ MemoryBlock3Iterator<T> end() {
    return MemoryBlock3Iterator<T>(data_, size_, pitch_, vec3i(-1));
  }

protected:
  vec3u size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
};

template <MemoryLocation L, typename T> class MemoryBlock3 {};

template <typename T> class MemoryBlock3<MemoryLocation::DEVICE, T> {
public:
  MemoryBlock3(const vec3u &size = vec3u()) : size_(size) {}
  void resize(const vec3u &size) { size_ = size; }
  ~MemoryBlock3() {
    if (data_)
      cudaFree(data_);
  }
  MemoryLocation location() const { return MemoryLocation::DEVICE; }
  const vec3u &size() const { return size_; }
  void allocate() {
    if (data_)
      cudaFree(data_);
    cudaPitchedPtr pdata;
    cudaExtent extent = make_cudaExtent(size_.x * sizeof(T), size_.y, size_.z);
    CUDA_CHECK(cudaMalloc3D(&pdata, extent));
    pitch_ = pdata.pitch;
    data_ = reinterpret_cast<T *>(pdata.ptr);
  }
  size_t memorySize() {
    return pitch_ * size_.x * size_.y * size_.z * sizeof(T);
  }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock3Accessor<T> accessor() {
    return MemoryBlock3Accessor<T>(data_, size_, pitch_);
  }
  cudaPitchedPtr pitchedData() {
    cudaPitchedPtr pd{0};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.x * sizeof(T);
    pd.ysize = size_.y;
    return pd;
  }

private:
  vec3u size_;
  size_t pitch_ = 0;
  T *data_ = nullptr;
};

template <typename T> class MemoryBlock3<MemoryLocation::HOST, T> {
public:
  MemoryBlock3(const vec3u &size = vec3u()) : size_(size) {}
  void resize(const vec3u &size) { size_ = size; }
  ~MemoryBlock3() {
    if (data_)
      delete[] data_;
  }
  MemoryLocation location() const { return MemoryLocation::HOST; }
  const vec3u &size() const { return size_; }
  void allocate() {
    if (data_)
      delete[] data_;
    pitch_ = size_.x * sizeof(T);
    data_ = new T[size_.x * size_.y * size_.z];
  }
  size_t memorySize() { return size_.x * size_.y * size_.z * sizeof(T); }
  T *ptr() { return data_; }
  const T *ptr() const { return data_; }
  MemoryBlock3Accessor<T> accessor() {
    return MemoryBlock3Accessor<T>(data_, size_, pitch_);
  }
  cudaPitchedPtr pitchedData() {
    cudaPitchedPtr pd{0};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.x * sizeof(T);
    pd.ysize = size_.y;
    return pd;
  }

private:
  vec3u size_;
  size_t pitch_ = 0;
  T *data_ = nullptr;
};

template <MemoryLocation A, MemoryLocation B, typename T>
bool memcpy(MemoryBlock1<A, T> &dst, MemoryBlock1<B, T> &src) {
  if (dst.size() != src.size())
    return false;
  auto kind = copyDirection(src.location(), dst.location());
  CUDA_CHECK(cudaMemcpy(dst.ptr(), src.ptr(), src.size() * sizeof(T), kind));
  return true;
}

template <MemoryLocation A, MemoryLocation B, typename T>
bool memcpy(MemoryBlock2<A, T> &dst, MemoryBlock2<B, T> &src) {
  if (dst.size() != src.size())
    return false;
  auto kind = copyDirection(src.location(), dst.location());
  copyPitchedToPitched<T>(dst.pitchedData(), src.pitchedData(), kind);
  return true;
}

template <MemoryLocation L, typename T>
bool memcpy(Array2<T> &dst, MemoryBlock2<L, T> &src) {
  if (dst.size() != src.size())
    return false;
  auto kind = copyDirection(src.location(), MemoryLocation::DEVICE);
  copyPitchedToArray<T>(dst.data(), src.pitchedData(), kind);
  return true;
}

template <MemoryLocation A, MemoryLocation B, typename T>
bool memcpy(MemoryBlock3<A, T> &dst, MemoryBlock3<B, T> &src) {
  if (dst.size() != src.size())
    return false;
  auto kind = copyDirection(src.location(), dst.location());
  copyPitchedToPitched<T>(dst.pitchedData(), src.pitchedData(), kind,
                          dst.size().z);
  return true;
}

template <MemoryLocation L, typename T>
bool memcpy(Array3<T> &dst, MemoryBlock3<L, T> &src) {
  if (dst.size() != src.size())
    return false;
  auto kind = copyDirection(src.location(), MemoryLocation::DEVICE);
  copyPitchedToArray<T>(dst.data(), src.pitchedData(), kind, dst.size().z);
  return true;
}

template <MemoryLocation L, typename T>
bool memcpy(std::vector<T> &dst, const MemoryBlock1<L, T> &src) {
  dst.resize(src.size());
  auto kind = copyDirection(src.location(), MemoryLocation::HOST);
  CUDA_CHECK(cudaMemcpy(dst.data(), src.ptr(), src.size() * sizeof(T), kind));
  return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock1<MemoryLocation::HOST, T> &data) {
  std::cerr << "1d MemoryBlock (" << data.size() << ")\n";
  auto acc = data.accessor();
  for (int x = 0; x < data.size(); x++) {
    os << "x[" << x << "]:\t";
    if (std::is_same<T, char>::value || std::is_same<T, unsigned char>::value)
      os << (int)acc[x] << "\t";
    else
      os << std::setprecision(6) << acc[x] << "\t";
    os << std::endl;
  }
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock1<MemoryLocation::DEVICE, T> &data) {
  MemoryBlock1<MemoryLocation::HOST, T> host(data.size());
  host.allocate();
  memcpy(host, data);
  os << host << std::endl;
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock2<MemoryLocation::HOST, T> &data) {
  std::cerr << "2d MemoryBlock (" << data.size().x << " x " << data.size().y
            << ")\n";
  auto acc = data.accessor();
  os << "x[" << -1 << "]:\t";
  for (int x = 0; x < data.size().x; x++)
    os << std::setw(10) << x << " ";
  os << std::endl;
  for (int y = data.size().y - 1; y >= 0; y--) {
    os << "y[" << y << "]:\t";
    for (int x = 0; x < data.size().x; x++)
      if (std::is_same<T, char>::value || std::is_same<T, unsigned char>::value)
        os << (int)acc(x, y) << "\t";
      else
        os << std::setprecision(6) << std::setw(10) << acc(x, y) << " ";
    os << std::endl;
  }
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock2<MemoryLocation::DEVICE, T> &data) {
  MemoryBlock2<MemoryLocation::HOST, T> host(data.size());
  host.allocate();
  memcpy(host, data);
  os << host << std::endl;
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock3<MemoryLocation::HOST, T> &data) {
  std::cerr << "3d MemoryBlock (" << data.size().x << " x " << data.size().y
            << " x " << data.size().z << ")\n";
  auto acc = data.accessor();
  os << "y     z  \\  x\t";
  for (int x = 0; x < data.size().x; x++)
    os << "      x[" << x << "]\t";
  os << std::endl;
  for (int z = 0; z < data.size().z; z++) {
    for (int y = data.size().y - 1; y >= 0; y--) {
      os << "y[" << y << "] z[" << z << "]:\t";
      for (int x = 0; x < data.size().x; x++)
        if (std::is_same<T, char>::value ||
            std::is_same<T, unsigned char>::value)
          os << (int)acc(x, y, z) << "\t";
        else
          os << std::setprecision(6) << std::setw(10) << acc(x, y, z) << "\t";
      os << std::endl;
    }
    os << "y     z  /  x\t";
    for (int x = 0; x < data.size().x; x++)
      os << "      x[" << x << "]\t";
    os << std::endl;
    os << "==================================================\n";
  }
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock3<MemoryLocation::DEVICE, T> &data) {
  MemoryBlock3<MemoryLocation::HOST, T> host(data.size());
  host.allocate();
  memcpy(host, data);
  os << host << std::endl;
  return os;
}
template <typename T>
__global__ void __fill2(MemoryBlock2Accessor<T> data, T value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  auto size = data.size();
  if (x < size.x && y < size.y)
    data(x, y) = value;
}

template <MemoryLocation L, typename T>
void fill2(MemoryBlock2<L, T> &data, T value);
template <typename T>
void fill2(MemoryBlock2<MemoryLocation::DEVICE, T> &data, T value) {
  ThreadArrayDistributionInfo td(data.size());
  __fill2<T><<<td.gridSize, td.blockSize>>>(data.accessor(), value);
}
template <typename T>
void fill2(MemoryBlock2<MemoryLocation::HOST, T> &data, T value) {
  for (auto e : data.accessor())
    e.value = value;
}

template <typename T>
__global__ void __fill3(MemoryBlock3Accessor<T> data, T value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  auto size = data.size();
  if (x < size.x && y < size.y && z < size.z)
    data(x, y, z) = value;
}

template <typename T> void fill3(MemoryBlock3Accessor<T> data, T value) {
  ThreadArrayDistributionInfo td(data.size());
  __fill3<T><<<td.gridSize, td.blockSize>>>(data, value);
}
template <typename T>
void fill3(MemoryBlock3<MemoryLocation::DEVICE, T> &data, T value) {
  ThreadArrayDistributionInfo td(data.size());
  __fill3<T><<<td.gridSize, td.blockSize>>>(data.accessor(), value);
}
template <typename T>
void fill3(MemoryBlock3<MemoryLocation::HOST, T> &data, T value) {
  for (auto e : data.accessor())
    e.value = value;
}

using MemoryBlock1Dd = MemoryBlock1<MemoryLocation::DEVICE, double>;
using MemoryBlock1Hd = MemoryBlock1<MemoryLocation::HOST, double>;
using MemoryBlock1Df = MemoryBlock1<MemoryLocation::DEVICE, float>;
using MemoryBlock1Hf = MemoryBlock1<MemoryLocation::HOST, float>;
using MemoryBlock1Di = MemoryBlock1<MemoryLocation::DEVICE, int>;
using MemoryBlock1Hi = MemoryBlock1<MemoryLocation::HOST, int>;
using MemoryBlock1Du = MemoryBlock1<MemoryLocation::DEVICE, unsigned int>;
using MemoryBlock1Hu = MemoryBlock1<MemoryLocation::HOST, unsigned int>;
using MemoryBlock2Df = MemoryBlock2<MemoryLocation::DEVICE, float>;
using MemoryBlock2Hf = MemoryBlock2<MemoryLocation::HOST, float>;
using MemoryBlock2Di = MemoryBlock2<MemoryLocation::DEVICE, int>;
using MemoryBlock2Hi = MemoryBlock2<MemoryLocation::HOST, int>;
using MemoryBlock2Duc = MemoryBlock2<MemoryLocation::DEVICE, unsigned char>;
using MemoryBlock2Huc = MemoryBlock2<MemoryLocation::HOST, unsigned char>;
using MemoryBlock2Du = MemoryBlock2<MemoryLocation::DEVICE, unsigned int>;
using MemoryBlock2Hu = MemoryBlock2<MemoryLocation::HOST, unsigned int>;
using MemoryBlock3Df = MemoryBlock3<MemoryLocation::DEVICE, float>;
using MemoryBlock3Hf = MemoryBlock3<MemoryLocation::HOST, float>;
using MemoryBlock3Di = MemoryBlock3<MemoryLocation::DEVICE, int>;
using MemoryBlock3Hi = MemoryBlock3<MemoryLocation::HOST, int>;
using MemoryBlock3Duc = MemoryBlock3<MemoryLocation::DEVICE, unsigned char>;
using MemoryBlock3Huc = MemoryBlock3<MemoryLocation::HOST, unsigned char>;
using MemoryBlock3Du = MemoryBlock3<MemoryLocation::DEVICE, unsigned int>;
using MemoryBlock3Hu = MemoryBlock3<MemoryLocation::HOST, unsigned int>;

} // namespace cuda
} // namespace hermes

#endif