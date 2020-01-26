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
///\file cuda_memory_block.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-25
///
///\brief

#ifndef HERMES_CUDA_MEMORY_BLOCK_H
#define HERMES_CUDA_MEMORY_BLOCK_H

#include <hermes/storage/cuda_storage_utils.h>

namespace hermes {
namespace cuda {

template <typename T> class CuMemoryBlock1Accessor {
public:
  CuMemoryBlock1Accessor(T *data, size_t size) : size_(size), data_(data) {}
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

/// Holds a linear memory area representing a 1-dimensional array.
/// \tparam T data type
template <typename T> class CuMemoryBlock1 {
public:
  CuMemoryBlock1() = default;
  CuMemoryBlock1(size_t size) : size_(size) { resize(size); }
  CuMemoryBlock1(const CuMemoryBlock1 &other) : CuMemoryBlock1(other.size_) {
    CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }
  CuMemoryBlock1(const CuMemoryBlock1 &&other) = delete;
  CuMemoryBlock1(CuMemoryBlock1 &&other) noexcept
      : size_(other.size_), data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
  }
  CuMemoryBlock1(const std::vector<T> &data) {
    resize(data.size());
    CHECK_CUDA(
        cudaMemcpy(data_, &data[0], size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~CuMemoryBlock1() { clear(); }
  ///\param other **[in]**
  ///\return CuMemoryBlock1<T>&
  CuMemoryBlock1<T> &operator=(const CuMemoryBlock1<T> &other) {
    resize(other.size_);
    CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    return *this;
  }
  ///\param data **[in]**
  ///\return CuMemoryBlock1<T>&
  CuMemoryBlock1<T> &operator=(const std::vector<T> &data) {
    resize(data.size());
    CHECK_CUDA(
        cudaMemcpy(data_, &data[0], size_ * sizeof(T), cudaMemcpyHostToDevice));
    return *this;
  }
  /// frees any previous data and allocates a new block
  ///\param new_size **[in]** new element count
  void resize(size_t new_size) {
    if (data_)
      clear();
    size_ = new_size;
    CHECK_CUDA(cudaMalloc(&data_, size_ * sizeof(T)));
  }
  ///\return size_t memory block size in elements
  size_t size() const { return size_; }
  ///\return size_t memory block size in bytes
  size_t memorySize() const { return size_ * sizeof(T); }
  ///\return const T* device pointer
  const T *ptr() const { return (const T *)data_; }
  ///\return  T* device pointer
  T *ptr() { return (T *)data_; }
  /// frees memory and set size to zero
  void clear() {
    if (data_)
      CHECK_CUDA(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
  }
  /// copies data to host side
  ///\return std::vector<T> data in host side
  std::vector<T> hostData() const {
    std::vector<T> data(size_);
    CHECK_CUDA(
        cudaMemcpy(&data[0], data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return data;
  }
  CuMemoryBlock1Accessor<T> accessor() {
    return CuMemoryBlock1Accessor<T>((T *)data_, size_);
  }

private:
  size_t size_{0};
  void *data_{nullptr};
};

using CuMemoryBlock1d = CuMemoryBlock1<double>;
using CuMemoryBlock1f = CuMemoryBlock1<float>;
using CuMemoryBlock1i = CuMemoryBlock1<int>;
using CuMemoryBlock1u = CuMemoryBlock1<unsigned int>;

} // namespace cuda

} // namespace hermes

#endif // HERMES_CUDA_MEMORY_BLOCK_H