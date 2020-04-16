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
///\file array.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-28
///
///\brief

#ifndef HERMES_STORAGE_ARRAY_H
#define HERMES_STORAGE_ARRAY_H

#include <hermes/common/index.h>
#include <hermes/storage/cuda_storage_utils.h>
#include <ponos/storage/array.h>

namespace hermes {

namespace cuda {

/*****************************************************************************
*******************           ARRAY1  ACCESSOR            ********************
******************************************************************************/
///\brief
///
///\tparam T
template <typename T> class Array1Accessor {
public:
  Array1Accessor(T *data, size_t size) : size_(size), data_(data) {}
  __host__ __device__ size_t size() const { return size_; }
  __device__ T &operator[](size_t i) { return data_[i]; }
  __device__ const T &operator[](size_t i) const { return data_[i]; }
  __device__ bool contains(int i) const { return i >= 0 && i < (int)size_; }

protected:
  size_t size_;
  T *data_ = nullptr;
};
/*****************************************************************************
*******************       ARRAY1 CONST ACCESSOR           ********************
******************************************************************************/
template <typename T> class Array1CAccessor {
public:
  Array1CAccessor(const T *data, size_t size) : size_(size), data_(data) {}
  __host__ __device__ size_t size() const { return size_; }
  __device__ const T &operator[](size_t i) const { return data_[i]; }
  __device__ bool contains(int i) const { return i >= 0 && i < (int)size_; }

protected:
  size_t size_;
  const T *data_ = nullptr;
};
/*****************************************************************************
**************************      ARRAY1 KERNELS       *************************
******************************************************************************/
template <typename T> __global__ void __fill(Array1Accessor<T> array, T value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (array.contains(x))
    array[x] = value;
}

template <typename T> void fill(Array1Accessor<T> array, T value) {
  ThreadArrayDistributionInfo td(array.size());
  __fill<T><<<td.gridSize, td.blockSize>>>(array, value);
}
/*****************************************************************************
**************************           ARRAY1          *************************
******************************************************************************/
/// Holds a linear memory area representing a 1-dimensional array.
/// \tparam T data type
template <typename T> class Array1 {
  static_assert(!std::is_same<T, bool>::value,
                "Array1 can't hold booleans, use char instead!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Array1() = default;
  Array1(size_t size) : size_(size) { resize(size); }
  Array1(size_t size, T value) : size_(size) {
    resize(size);
    auto acc = Array1Accessor<T>((T *)data_, size_);
    fill(acc, value);
  }
  Array1(const Array1 &other) : Array1(other.size_) {
    CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }
  Array1(const Array1 &&other) = delete;
  Array1(Array1 &&other) noexcept : size_(other.size_), data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
  }
  Array1(const std::vector<T> &data) {
    resize(data.size());
    CHECK_CUDA(
        cudaMemcpy(data_, &data[0], size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Array1() { clear(); }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  ///\param other **[in]**
  ///\return Array1<T>&
  Array1<T> &operator=(const Array1<T> &other) {
    resize(other.size_);
    CHECK_CUDA(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    return *this;
  }
  ///\param data **[in]**
  ///\return Array1<T>&
  Array1<T> &operator=(const std::vector<T> &data) {
    resize(data.size());
    CHECK_CUDA(
        cudaMemcpy(data_, &data[0], size_ * sizeof(T), cudaMemcpyHostToDevice));
    return *this;
  }
  Array1<T> &operator=(T value) {
    fill(Array1Accessor<T>((T *)data_, size_), value);
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
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
  const T *data() const { return (const T *)data_; }
  ///\return  T* device pointer
  T *data() { return (T *)data_; }
  /// copies data to host side
  ///\return std::vector<T> data in host side
  std::vector<T> hostData() const {
    std::vector<T> h_data(size_);
    CHECK_CUDA(cudaMemcpy(&h_data[0], data_, size_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return h_data;
  }
  Array1Accessor<T> accessor() { return Array1Accessor<T>((T *)data_, size_); }
  Array1CAccessor<T> constAccessor() const {
    return Array1CAccessor<T>((const T *)data_, size_);
  }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  /// frees memory and set size to zero
  void clear() {
    if (data_)
      CHECK_CUDA(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
  }

private:
  u32 size_{0};
  void *data_{nullptr};
};

using array1d = Array1<f64>;
using array1f = Array1<f32>;
using array1i = Array1<i32>;
using array1u = Array1<u32>;

/*****************************************************************************
*************************       ARRAY2 ACCESSOR      *************************
******************************************************************************/
///\tparam T array data type
template <typename T> class Array2Accessor {
public:
  Array2Accessor(void *data, size2 size, size_t pitch)
      : size_(size), pitch_(pitch), data_(data) {}
  __host__ __device__ size2 size() const { return size_; }
  __device__ T &operator[](index2 ij) {
    return (T &)(*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  __device__ bool contains(index2 ij) const {
    return ij >= index2() && ij < size_;
  }

protected:
  size2 size_;
  size_t pitch_;
  void *data_ = nullptr;
};
/*****************************************************************************
*********************     ARRAY2 CONST ACCESSOR      *************************
******************************************************************************/
///\tparam T array data type
template <typename T> class Array2CAccessor {
public:
  Array2CAccessor(const void *data, size2 size, size_t pitch)
      : size_(size), pitch_(pitch), data_(data) {}
  __host__ __device__ size2 size() const { return size_; }
  __device__ T operator[](index2 ij) const {
    return (T &)*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T));
  }
  __device__ bool contains(index2 ij) const {
    return ij >= index2() && ij < size_;
  }

protected:
  size2 size_;
  size_t pitch_;
  const void *data_ = nullptr;
};
/*****************************************************************************
**************************      ARRAY2 KERNELS       *************************
******************************************************************************/
template <typename T> __global__ void __fill(Array2Accessor<T> array, T value) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (array.contains(index))
    array[index] = value;
}
/// Ste all values of the array2 to **value**
///\tparam T array data type
///\param array **[in]**
///\param value **[in]**
template <typename T> void fill(Array2Accessor<T> array, T value) {
  ThreadArrayDistributionInfo td(array.size());
  __fill<T><<<td.gridSize, td.blockSize>>>(array, value);
}
/// \tparam T
/// \param destination **[in]**
/// \param source **[in]**
template <typename T>
__global__ void __copy(Array2Accessor<T> destination,
                       Array2CAccessor<T> source) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (destination.contains(index) && source.contains(index))
    destination[index] = source[index];
}
/// \tparam T
/// \param destination **[in]**
/// \param source **[in]**
template <typename T>
void copy(Array2Accessor<T> destination, Array2CAccessor<T> source) {
  ThreadArrayDistributionInfo td(destination.size());
  __copy<T><<<td.gridSize, td.blockSize>>>(destination, source);
}
/// Auxiliary class that encapsulates a c++ lambda function into device code
///\tparam T array data type
///\tparam F lambda function type following the signature: (index2, T&)
template <typename T, typename F> struct map_operation {
  __host__ __device__ explicit map_operation(const F &op) : operation(op) {}
  __host__ __device__ void operator()(index2 index, T &value) {
    operation(index, value);
  }
  F operation;
};
///\tparam T
///\tparam F
///\param array **[in]**
///\param operation **[in]**
template <typename T, typename F>
__global__ void __map(Array2Accessor<T> array, map_operation<T, F> operation) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (array.contains(index))
    operation(index, array[index]);
}
/// Apply **operation** to all elements of **array**
///\tparam T array data type
///\tparam F lambda function type
///\param array **[in]**
///\param operation **[in]** lambda function with the following signature:
///(index2, T&)
template <typename T, typename F>
void __host__ __device__ mapArray(Array2Accessor<T> array,
                                  map_operation<T, F> operation) {
  ThreadArrayDistributionInfo td(array.size());
  __map<T, F><<<td.gridSize, td.blockSize>>>(array, operation);
}
/*****************************************************************************
*************************           ARRAY2           *************************
******************************************************************************/
/// Holds a linear memory area representing a 2-dimensional array of
/// ``size.width`` * ``size.height`` elements.
///
/// - Considering ``size.height`` rows of ``size.width`` elements, data is
/// laid out in memory in **row major** fashion.
///
/// - Elements must be accessed through an Array2Accessor object, **in device
/// code**.
///
/// - The methods use the convention of ``i`` and ``j`` indices, representing
/// _column_ and _row_ indices respectively. ``i`` accesses the first
/// dimension (``size.width``) and ``j`` accesses the second dimension
/// (``size.height``).
/// \verbatim embed:rst:leading-slashes"
///   .. note::
///     This index convention is the **opposite** of some mathematical forms
///     where matrix elements are indexed by the i-th row and j-th column.
///   .. note::
///     This class is the equivalent of ponos's Array2. They can be user
///     interchangeably to work with data in host and device.
/// \endverbatim
/// \tparam T data type
template <typename T> class Array2 {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Array2() = default;
  ///\param size **[in]** dimensions (in elements count)
  Array2(size2 size) : size_(size) { resize(size); }
  /// Constructor
  ///\param size **[in]** dimensions (in elements count)
  ///\param value **[in]** initial value of all elements
  Array2(size2 size, T value) : size_(size) {
    resize(size);
    auto acc = Array2Accessor<T>((T *)data_, size_, pitch_);
    fill(acc, value);
  }
  /// Copy constructor
  ///\param other **[in]** const reference to other Array2 object
  Array2(const Array2<T> &other) {
    resize(other.size_);
    copyPitchedToPitched<T>(pitchedData(), other.pitchedData(),
                            cudaMemcpyDeviceToDevice);
  }
  /// Assign constructor
  ///\param other **[in]** temporary Array2 object
  Array2(Array2 &&other) noexcept
      : size_(other.size_), data_(other.data_), pitch_(other.pitch_) {
    other.size_ = size2(0, 0);
    other.pitch_ = 0;
    other.data_ = nullptr;
  }
  Array2(const ponos::Array2<T> &data) = delete;
  /// Constructor from host data
  ///\param data **[in]** reference to host Array2 object
  Array2(ponos::Array2<T> &data) {
    resize(size2(data.size()));
    copyPitchedToPitched<T>(pitchedData(), pitchedDataFrom(data),
                            cudaMemcpyHostToDevice);
  }
  ~Array2() { clear(); }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// Assign operator
  /// \verbatim embed:rst:leading-slashes"
  ///   .. warning::
  ///     Copying from const objects is currently done using kernels.
  /// \endverbatim
  ///\param other **[in]** const reference to other Array2 object
  ///\return Array2<T>&
  Array2<T> &operator=(const Array2<T> &other) {
    resize(other.size());
    if (size_.total() > 0)
      copy((*this).accessor(), other.constAccessor());
    return *this;
  }
  /// Assign operator
  ///\param other **[in]** reference to other Array 2 object
  ///\return Array1<T>&
  Array2<T> &operator=(Array2<T> &other) {
    resize(other.size_);
    if (size_.total() > 0)
      copyPitchedToPitched<T>(pitchedData(), other.pitchedData(),
                              cudaMemcpyDeviceToDevice);
    return *this;
  }
  Array2<T> &operator=(const ponos::Array2<T> &data) = delete;
  Array2<T> &operator=(const ponos::Array2<T> &&data) = delete;
  /// Assign operator from host data
  ///\param data **[in]** host data
  ///\return Array1<T>&
  Array2<T> &operator=(ponos::Array2<T> &data) {
    resize(size2(data.size().width, data.size().height));
    if (data.size().total() > 0)
      copyPitchedToPitched<T>(pitchedData(), pitchedDataFrom(data),
                              cudaMemcpyHostToDevice);
    return *this;
  }
  /// Assign operator from host data
  /// \param data **[in]** host data temporary object
  /// \return Array2<T>&
  Array2<T> &operator=(ponos::Array2<T> &&data) {
    resize(size2(data.size().width, data.size().height));
    if (size_.total() > 0)
      copyPitchedToPitched<T>(pitchedData(), pitchedData(data),
                              cudaMemcpyHostToDevice);
    return *this;
  }
  /// Assigns ``value`` to all elements
  /// \param value **[in]**
  /// \return Array2<T>&
  Array2<T> &operator=(T value) {
    if (size_.total() > 0)
      fill(Array2Accessor<T>((T *)data_, size_, pitch_), value);
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \verbatim embed:rst:leading-slashes"
  ///   .. warning::
  ///     This method frees any previous data and allocates a new block.
  /// \endverbatim
  ///\param new_size **[in]** new dimensions (in elements count)
  void resize(size2 new_size) {
    if (data_)
      clear();
    size_ = new_size;
    if (size_.total() == 0) {
      clear();
      return;
    }
    void *pdata = nullptr;
    CHECK_CUDA(cudaMallocPitch(&pdata, &pitch_, size_.width * sizeof(T),
                               size_.height));
    data_ = reinterpret_cast<T *>(pdata);
  }
  ///\return size_t memory block size in elements
  size2 size() const { return size_; }
  ///\return u32 pitch size (in bytes)
  size_t pitch() const { return pitch_; }
  ///\return size_t memory block size in bytes
  size_t memorySize() const { return pitch_ * size_.height; }
  ///\return const T* raw device pointer
  const T *data() const { return (const T *)data_; }
  ///\return  T* raw device pointer
  T *data() { return (T *)data_; }
  ///\return cudaPitchedPtr
  cudaPitchedPtr pitchedData() const {
    cudaPitchedPtr pd{0};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.width * sizeof(T);
    pd.ysize = size_.height;
    return pd;
  }
  /// copies data to host side
  ///\return std::vector<T> data in host side
  ponos::Array2<T> hostData() {
    ponos::Array2<T> data(ponos::size2(size_.width, size_.height), pitch_);
    copyPitchedToPitched<T>(pitchedDataFrom(data), pitchedData(),
                            cudaMemcpyDeviceToHost);
    return data;
  }
  /// \return Array2Accessor<T>
  Array2Accessor<T> accessor() {
    return Array2Accessor<T>(data_, size_, pitch_);
  }
  /// \return Array2CAccessor<T>
  Array2CAccessor<T> constAccessor() const {
    return Array2CAccessor<T>(data_, size_, pitch_);
  }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  /// frees memory and set size to zero
  void clear() {
    if (data_)
      CHECK_CUDA(cudaFree(data_));
    data_ = (T *)nullptr;
    size_ = size2(0, 0);
  }
  /// Applies ``operation`` to all elements
  ///\tparam F function type with an ``operator()`` following the signature:
  ///``(index2, T&)``
  ///\param operation **[in]**
  template <typename F> void map(F operation) {
    mapArray<T, F>(Array2Accessor<T>((T *)data_, size_, pitch_),
                   map_operation<T, F>(operation));
  }

private:
  size2 size_{0};
  size_t pitch_ = 0;
  void *data_{nullptr};
};

template <typename T>
std::ostream &operator<<(std::ostream &os, Array2<T> &array) {
  auto h_array = array.hostData();
  os << h_array;
  return os;
}

using array2d = Array2<f64>;
using array2f = Array2<f32>;
using array2i = Array2<i32>;
using array2u = Array2<u32>;
} // namespace cuda

} // namespace hermes

#endif