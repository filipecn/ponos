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
/// \date 2020-02-28
///
/// \brief

#ifndef PONOS_NUMERIC_FD_MATRIX_H
#define PONOS_NUMERIC_FD_MATRIX_H

#include <ponos/blas/vector.h>
#include <ponos/storage/array.h>

namespace ponos {

/*****************************************************************************
**************************         FDMATRIX2         *************************
******************************************************************************/
template <typename T> struct FDMatrix2Entry { T diag, x, y; };
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
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  FDMatrix2 &operator=(const FDMatrix2 &other) {
    A_ = other.data();
    indices_ = other.indexData();
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  ///
  /// \param l **[in]**
  /// \param c **[in]**
  /// \return bool
  bool stores(index2 l, index2 c) const {
    if (!A_.stores(l))
      return false;
    if (indices_[l] < 0)
      return false;
    if (distance(l, c) > 1)
      return false;
    if (!indices_.stores(c))
      return false;
    if (indices_[c] < 0)
      return false;
    return true;
  }
  ///
  /// \param l **[in]**
  /// \param c **[in]**
  /// \return T&
  T &operator()(index2 l, index2 c) {
    dummy_ = 0.f;
    if (!indices_.stores(c))
      return dummy_;
    if (indices_[c] < 0)
      return dummy_;
    if (l == c && A_.stores(l) && indices_[l] >= 0)
      return A_[l].diag;
    if (c == l.right() && A_.stores(l) && indices_[l] >= 0)
      return A_[l].x;
    if (c.right() == l && A_.stores(c) && indices_[c] >= 0)
      return A_[c].x;
    if (c == l.up() && A_.stores(l) && indices_[l] >= 0)
      return A_[l].y;
    if (c.up() == l && A_.stores(c) && indices_[c] >= 0)
      return A_[c].y;
    return dummy_;
  }
  ///
  /// \param l **[in]**
  /// \param c **[in]**
  /// \return T&
  T operator()(index2 l, index2 c) const {
    if (!indices_.stores(c))
      return dummy_;
    if (indices_[c] < 0)
      return 0;
    if (l == c && A_.stores(l) && indices_[l] >= 0)
      return A_[l].diag;
    if (c == l.right() && A_.stores(l) && indices_[l] >= 0)
      return A_[l].x;
    if (c.right() == l && A_.stores(c) && indices_[c] >= 0)
      return A_[c].x;
    if (c == l.up() && A_.stores(l) && indices_[l] >= 0)
      return A_[l].y;
    if (c.up() == l && A_.stores(c) && indices_[c] >= 0)
      return A_[c].y;
    return 0;
  }
  /// \brief
  ///
  /// \param ij **[in]**
  /// \return int
  int elementIndex(index2 ij) const {
    if (!indices_.stores(ij))
      return -1;
    return indices_[ij];
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
  Array2<FDMatrix2Entry<T>> &data() { return A_; }
  /// \return Array2<i32>&
  Array2<i32> &indexData() { return indices_; }
  /// \return const Array2<FDMatrix2Entry>&
  const Array2<FDMatrix2Entry<T>> &data() const { return A_; }
  /// \return const Array2<i32>&
  const Array2<i32> &indexData() const { return indices_; }

private:
  T dummy_;
  Array2<FDMatrix2Entry<T>> A_;
  Array2<i32> indices_;
};
// ***********************************************************************
//                            ARITHMETIC
// ***********************************************************************
template <typename T>
Vector<T> operator*(const FDMatrix2<T> &A, const Vector<T> &x) {
  Vector<T> b(x.size());
  for (index2 index : Index2Range<i32>(A.gridSize())) {
    if (!A.stores(index, index))
      continue;
    int i = A.elementIndex(index);
    if (i < 0)
      continue;
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
  return b;
}

} // namespace ponos

#endif