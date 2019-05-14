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
#include <iomanip>

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

template <typename T> class MemoryBlock3Accessor {
public:
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

protected:
  vec3u size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
};

template <MemoryLocation L, typename T> class MemoryBlock3 {
  // public:
  //   MemoryBlock3(const vec3u &size = vec3u()) : size_(size) {}
  //   MemoryLocation location() const { return L; }
  //   void resize(const vec3u &size) { size_ = size; }
  //   size_t dataTypeSize() { return sizeof(T); };
  //   virtual __host__ __device__ const vec3u &size() const { return size_; }
  //   virtual void allocate() = 0;
  //   virtual size_t memorySize() = 0;
  //   virtual T *ptr() = 0;
  //   virtual const T *ptr() const = 0;

  // protected:
  //   vec3u size_;
};

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

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         MemoryBlock3<MemoryLocation::HOST, T> &data) {
  std::cerr << "3d MemoryBlock (" << data.size().x << " x " << data.size().y
            << " x " << data.size().z << ")\n";
  auto acc = data.accessor();
  for (int z = 0; z < data.size().z; z++)
    for (int y = data.size().y - 1; y >= 0; y--) {
      os << "l[" << y << "] d[" << z << "]: ";
      for (int x = 0; x < data.size().z; x++)
        if (std::is_same<T, char>::value ||
            std::is_same<T, unsigned char>::value)
          os << (int)acc(x, y, z) << "\t";
        else
          os << std::setprecision(6) << acc(x, y, z) << "\t";
      os << std::endl;
    }
  return os;
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

} // namespace cuda
} // namespace hermes

#endif