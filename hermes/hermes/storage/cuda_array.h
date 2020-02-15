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

#ifndef HERMES_STORAGE_CUDA_ARRAY_H
#define HERMES_STORAGE_CUDA_ARRAY_H

#include <hermes/common/cuda.h>

namespace hermes {

namespace cuda {

template <typename T> class CuArray2 {
public:
  CuArray2(hermes::cuda::vec2u size) : size_(size) { allocate(); }
  ~CuArray2() {
    if (array_)
      cudaFreeArray(array_);
  }
  hermes::cuda::vec2u size() const { return size_; }
  void allocate() {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CHECK_CUDA(cudaMallocArray(&array_, &channelDesc, size_.x, size_.y));
  }
  const cudaArray *data() const { return array_; }
  cudaArray *data() { return array_; }

private:
  hermes::cuda::vec2u size_;
  cudaArray *array_ = nullptr;
};

template <typename T> class CuArray3 {
public:
  CuArray3(hermes::cuda::vec3u size) : size_(size) { allocate(); }
  ~CuArray3() {
    if (array_)
      cudaFreeArray(array_);
  }
  hermes::cuda::vec3u size() const { return size_; }
  void allocate() {
    cudaExtent extent = make_cudaExtent(size_.x, size_.y, size_.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CHECK_CUDA(cudaMalloc3DArray(&array_, &channelDesc, extent));
  }
  const cudaArray *data() const { return array_; }
  cudaArray *data() { return array_; }

private:
  hermes::cuda::vec3u size_;
  cudaArray *array_ = nullptr;
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_ARRAY_H