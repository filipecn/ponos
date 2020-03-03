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

#include <hermes/storage/array.h>

namespace hermes {

namespace cuda {

/*****************************************************************************
************************           CUARRAY2          *************************
******************************************************************************/
template <typename T> class CuArray2 {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  CuArray2() = default;
  CuArray2(size2 size) : size_(size) { resize(size_); }
  ~CuArray2() {
    if (array_)
      cudaFreeArray(array_);
  }
  CuArray2(Array2<T> &array) {
    resize(array.size());
    copyPitchedToArray<T>(array_, array.pitchedData(),
                          cudaMemcpyDeviceToDevice);
  }
  CuArray2(ponos::Array2<T> &array) {
    resize(size2(array.size().width, array.size().height));
    copyPitchedToArray<T>(array_, pitchedDataFrom(array),
                          cudaMemcpyHostToDevice);
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  size2 size() const { return size_; }
  void resize(size2 new_size) {
    size_ = new_size;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CHECK_CUDA(
        cudaMallocArray(&array_, &channelDesc, size_.width, size_.height));
  }
  const cudaArray *data() const { return array_; }
  cudaArray *data() { return array_; }

private:
  size2 size_{};
  cudaArray *array_ = nullptr;
}; // namespace cuda

/*****************************************************************************
************************           CUARRAY3          *************************
******************************************************************************/
template <typename T> class CuArray3 {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  CuArray3(size3 size) : size_(size) { resize(size_); }
  ~CuArray3() { clear(); }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  size3 size() const { return size_; }
  void resize(size3 new_size) {
    clear();
    size_ = new_size;
    cudaExtent extent = make_cudaExtent(size_.width, size_.height, size_.depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CHECK_CUDA(cudaMalloc3DArray(&array_, &channelDesc, extent));
  }
  const cudaArray *data() const { return array_; }
  cudaArray *data() { return array_; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  void clear() {
    if (array_)
      cudaFreeArray(array_);
  }

private:
  size3 size_{};
  cudaArray *array_ = nullptr;
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_ARRAY_H