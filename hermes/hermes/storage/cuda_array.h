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

#include <cuda_runtime.h>
#include <hermes/common/cuda_parallel.h>
#include <hermes/common/defs.h>
#include <hermes/geometry/cuda_vector.h>
#include <hermes/storage/cuda_array_kernels.h>

namespace hermes {

namespace cuda {

/// Accessor for arrays stored on the device.
/// Indices are accessed as: i * width * height + j * height + k
/// \tparam T data type
template <typename T> class Array3 {
public:
  /// \param size array size
  /// \param data raw pointer to device data
  /// \param accessMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  __host__ __device__ Array3(vec3u size, T *data,
                             AddressMode accessMode = AddressMode::NONE,
                             T border = T())
      : size(size), dataSize(size[0] * size[1] * size[2]), data(data),
        accessMode(accessMode), border(border) {}
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ T &operator()(int i, int j, int k) {
    int index = size[1] * (i * size[0] + j) + k;
    if (i >= 0 && i < size[0] && j >= 0 && j < size[1] && k >= 0 && k < size[2])
      return data[index];
    return dummy;
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return const T& const reference to data (out of borders data is follows
  /// the access mode)
  __host__ __device__ const T &operator()(int i, int j, int k) const {
    if (i >= 0 && i < size[0] && j >= 0 && j < size[1] && k >= 0 &&
        k < size[2]) {
      int index = size[1] * (i * size[0] + j) + k;
      return data[index];
    }
    if (accessMode == AddressMode::NONE)
      return dummy;
    if (accessMode == AddressMode::BORDER)
      return border;
    // if (accessMode == AccessMode::CLAMP_TO_EDGE) {
    i = std::max(0, std::min(i, static_cast<int>(size[0]) - 1));
    j = std::max(0, std::min(j, static_cast<int>(size[1]) - 1));
    k = std::max(0, std::min(k, static_cast<int>(size[2]) - 1));
    // }
    int index = size[1] * (i * size[0] + j) + k;
    return data[index];
  }
  /// \param index 1-dimensional index (follows i * width * height + j * height
  /// + k)
  /// \return T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ T &operator[](int index) {
    if (index >= 0 && index < dataSize)
      return data[index];
    return dummy;
  }
  /// Set all data to value
  /// \param value
  __host__ void fill(T value) {
    ThreadArrayDistributionInfo td(dataSize);
    // __fillArray<T><<<td.gridSize, td.blockSize>>>(data, dataSize, value);
  }

private:
  vec3u size;             //!< array size
  size_t dataSize;        //!< contiguous array size
  T *data = nullptr;      //!< raw pointer to device data
  AddressMode accessMode; //!< defines how out of bounds data is treated
  T border;               //!< border value
  T dummy;                //!< used as out of bounds reference variable
};

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_ARRAY_H