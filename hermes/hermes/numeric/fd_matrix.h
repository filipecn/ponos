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
/// \file fd_matrix.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-02-29
///
/// \brief

#ifndef HERMES_NUMERIC_FD_MATRIX_H
#define HERMES_NUMERIC_FD_MATRIX_H

#include <hermes/blas/vector.h>
#include <ponos/numeric/fd_matrix.h>

namespace hermes {

namespace cuda {

/*****************************************************************************
********************         FDMATRIX2 Accessor         **********************
******************************************************************************/
template <typename T> class FDMatrix2Accessor {
public:
  FDMatrix2Accessor(Array2Accessor<ponos::FDMatrix2Entry<T>> A,
                    Array2CAccessor<int> indices)
      : A_(A), indices_(indices) {}
  __host__ __device__ size2 gridSize() const { return A_.size(); }
  __host__ __device__ size_t size() const { return A_.size().total(); }
  __device__ bool stores(index2 l, index2 c) const {
    if (!A_.contains(l))
      return false;
    if (indices_[l] < 0)
      return false;
    if (distance(l, c) > 1)
      return false;
    if (!indices_.contains(c))
      return false;
    if (indices_[c] < 0)
      return false;
    return true;
  }
  __device__ T &operator()(index2 l, index2 c) {
    dummy = 0.f;
    if (!indices_.contains(c))
      return dummy;
    if (indices_[c] < 0)
      return dummy;
    if (l == c && A_.contains(l) && indices_[l] >= 0)
      return A_[l].diag;
    if (c == l.right() && A_.contains(l) && indices_[l] >= 0)
      return A_[l].x;
    if (c.right() == l && A_.contains(c) && indices_[c] >= 0)
      return A_[c].x;
    if (c == l.up() && A_.contains(l) && indices_[l] >= 0)
      return A_[l].y;
    if (c.up() == l && A_.contains(c) && indices_[c] >= 0)
      return A_[c].y;
    return dummy;
  }
  __device__ int elementIndex(index2 ij) const {
    if (!indices_.contains(ij))
      return -1;
    return indices_[ij];
  }

private:
  T dummy;
  Array2Accessor<ponos::FDMatrix2Entry<T>> A_;
  Array2CAccessor<int> indices_;
};

/*****************************************************************************
****************         FDMATRIX2 Const Accessor       **********************
******************************************************************************/
template <typename T> class FDMatrix2CAccessor {
public:
  FDMatrix2CAccessor(Array2CAccessor<ponos::FDMatrix2Entry<T>> A,
                     Array2CAccessor<int> indices)
      : A_(A), indices_(indices) {}
  __host__ __device__ size2 gridSize() const { return A_.size(); }
  __host__ __device__ size_t size() const { return A_.size().total(); }
  __device__ bool stores(index2 l, index2 c) const {
    if (!A_.contains(l))
      return false;
    if (indices_[l] < 0)
      return false;
    if (distance(l, c) > 1)
      return false;
    if (!indices_.contains(c))
      return false;
    if (indices_[c] < 0)
      return false;
    return true;
  }
  __device__ T operator()(index2 l, index2 c) const {
    if (!indices_.contains(c))
      return 0;
    if (indices_[c] < 0)
      return 0;
    if (l == c && A_.contains(l) && indices_[l] >= 0)
      return A_[l].diag;
    if (c == l.right() && A_.contains(l) && indices_[l] >= 0)
      return A_[l].x;
    if (c.right() == l && A_.contains(c) && indices_[c] >= 0)
      return A_[c].x;
    if (c == l.up() && A_.contains(l) && indices_[l] >= 0)
      return A_[l].y;
    if (c.up() == l && A_.contains(c) && indices_[c] >= 0)
      return A_[c].y;
    return 0;
  }
  __device__ int elementIndex(hermes::cuda::index2 ij) const {
    if (!indices_.contains(ij))
      return -1;
    return indices_[ij];
  }

private:
  Array2CAccessor<ponos::FDMatrix2Entry<T>> A_;
  Array2CAccessor<int> indices_;
};

/*****************************************************************************
**************************         FDMATRIX2         *************************
******************************************************************************/
/// Compact representation of a 3 band symmetric matrix built from a numerical
/// grid
template <typename T> class FDMatrix2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "FDMatrix2 must hold an float type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  FDMatrix2() = default;
  ///
  /// \param size **[in]**
  FDMatrix2(const size2 &size) { resize(size); }
  FDMatrix2(ponos::FDMatrix2<T> &ponos_other) {
    A_ = ponos_other.data();
    indices_ = ponos_other.indexData();
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  FDMatrix2 &operator=(const FDMatrix2 &other) {
    A_ = other.data();
    indices_ = other.indexData();
    return *this;
  }
  FDMatrix2 &operator=(const ponos::FDMatrix2<T> &ponos_other) {
    A_ = ponos_other.data();
    indices_ = ponos_other.indexData();
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  ponos::FDMatrix2<T> hostData() {
    ponos::FDMatrix2<T> h;
    h.data() = A_.hostData();
    h.indexData() = indices_.hostData();
    return h;
  }
  /// \param size **[in]**
  void resize(const size2 &size) {
    A_.resize(size);
    indices_.resize(size);
  }
  /// \return size2
  size2 gridSize() const { return A_.size(); }
  /// \return size_t
  size_t size() const { return A_.size().total(); }

  /// \return Array2<FDMatrix2Entry>&
  Array2<ponos::FDMatrix2Entry<T>> &data() { return A_; }
  /// \return Array2<i32>&
  Array2<i32> &indexData() { return indices_; }
  /// \return const Array2<FDMatrix2Entry>&
  const Array2<ponos::FDMatrix2Entry<T>> &data() const { return A_; }
  /// \return const Array2<i32>&
  const Array2<i32> &indexData() const { return indices_; }
  /// \return FDMatrix2Accessor<T>
  FDMatrix2Accessor<T> accessor() {
    return FDMatrix2Accessor<T>(A_.accessor(), indices_.constAccessor());
  }
  /// \return FDMatrix2CAccessor<T>
  FDMatrix2CAccessor<T> constAccessor() const {
    return FDMatrix2CAccessor<T>(A_.constAccessor(), indices_.constAccessor());
  }

private:
  Array2<ponos::FDMatrix2Entry<T>> A_;
  Array2<i32> indices_;
};
// ***********************************************************************
//                            ARITHMETIC
// ***********************************************************************

template <typename T>
__global__ void __mul(FDMatrix2CAccessor<T> A, Array1CAccessor<T> x,
                      Array1Accessor<T> b) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (!A.stores(index, index))
    return;
  int i = A.elementIndex(index);
  if (i < 0)
    return;
  b[i] = A(index, index) * x[i];
  int idx = A.elementIndex(index.left());
  if (idx >= 0)
    b[i] += A(index, index.left()) * x[idx];
  idx = A.elementIndex(index.right());
  if (idx >= 0)
    b[i] += A(index, index.right()) * x[idx];
  idx = A.elementIndex(index.down());
  if (idx >= 0)
    b[i] += A(index, index.down()) * x[idx];
  idx = A.elementIndex(index.up());
  if (idx >= 0)
    b[i] += A(index, index.up()) * x[idx];
}

template <typename T>
Vector<T> operator*(const FDMatrix2<T> &A, const Vector<T> &x) {
  Vector<T> b(x.size());
  hermes::cuda::ThreadArrayDistributionInfo td(A.gridSize());
  __mul<<<td.gridSize, td.blockSize>>>(
      A.constAccessor(), x.data().constAccessor(), b.data().accessor());
  return b;
}

} // namespace cuda

} // namespace hermes

#endif