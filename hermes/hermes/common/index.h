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
///\file index.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-28
///
///\brief

#ifndef HERMES_COMMON_INDEX_H
#define HERMES_COMMON_INDEX_H

#include <hermes/common/size.h>
#include <hermes/geometry/cuda_numeric.h>
#include <ponos/common/index.h>

namespace hermes {

namespace cuda {

/*****************************************************************************
**************************          INDEX2           *************************
******************************************************************************/
/// Holds 2-dimensional index coordinates
///\tparam T must be an integer type
template <typename T> struct Index2 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index2 must hold an integer type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  ///\brief Construct a new Index2 object
  ///\param i **[in]** i coordinate value
  ///\param j **[in]** j coordinate value
  Index2() = default;
  __host__ __device__ explicit Index2(T ij) : i(ij), j(ij) {}
  __host__ __device__ explicit Index2(T i, T j) : i(i), j(j) {}
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  __host__ __device__ T operator[](int _i) const { return (&i)[_i]; }
  __host__ __device__ T &operator[](int _i) { return (&i)[_i]; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  __host__ __device__ Index2<T> plus(T _i, T _j) const {
    return Index2<T>(i + _i, j + _j);
  }
  __host__ __device__ Index2<T> left() const { return Index2<T>(i - 1, j); }
  __host__ __device__ Index2<T> right() const { return Index2<T>(i + 1, j); }
  __host__ __device__ Index2<T> up() const { return Index2<T>(i, j + 1); }
  __host__ __device__ Index2<T> down() const { return Index2<T>(i, j - 1); }
  __host__ __device__ void clampTo(const size2 &s) {
    i = max(0, min(i, static_cast<T>(s.width)));
    j = max(0, min(j, static_cast<T>(s.height)));
  }
  ponos::Index2<T> ponos() const { return ponos::Index2<T>(i, j); }
  // ***********************************************************************
  //                            FIELDS
  // ***********************************************************************
  T i = T(0);
  T j = T(0);
};
// ***********************************************************************
//                             ARITHMETIC
// ***********************************************************************
template <typename T>
__host__ __device__ Index2<T> operator+(const Index2<T> &a,
                                        const Index2<T> &b) {
  return Index2<T>(a.i + b.i, a.j + b.j);
}
template <typename T>
__host__ __device__ Index2<T> operator-(const Index2<T> &a,
                                        const Index2<T> &b) {
  return Index2<T>(a.i - b.i, a.j - b.j);
}
/// \brief Computes the manhattan distance between two indices
/// \tparam T
/// \param a **[in]**
/// \param b **[in]**
/// \return T
template <typename T>
__host__ __device__ T distance(const Index2<T> &a, const Index2<T> &b) {
  return fabsf(a.i - b.i) + fabsf(a.j - b.j);
}
// ***********************************************************************
//                             BOOLEAN
// ***********************************************************************
template <typename T>
__host__ __device__ bool operator<=(const Index2<T> &a, const Index2<T> &b) {
  return a.i <= b.i && a.j <= b.j;
}
template <typename T>
__host__ __device__ bool operator>=(const Index2<T> &a, const Index2<T> &b) {
  return a.i >= b.i && a.j >= b.j;
}
///\brief are equal? operator
///\param other **[in]**
///\return bool true if both coordinate values are equal
template <typename T>
__host__ __device__ bool operator==(const Index2<T> &a, const Index2<T> &b) {
  return a.i == b.i && a.j == b.j;
}
/// \brief are different? operator
///\param other **[in]**
///\return bool true if any coordinate value is different
template <typename T>
__host__ __device__ bool operator!=(const Index2<T> &a, const Index2<T> &b) {
  return a.i != b.i || a.j != b.j;
}
template <typename T, typename TT>
__host__ __device__ bool operator<(const Index2<T> &a, const Size2<TT> &b) {
  return a.i < static_cast<T>(b.width) && a.j < static_cast<T>(b.height);
}
template <typename T, typename TT>
__host__ __device__ bool operator>(const Index2<T> &a, const Size2<TT> &b) {
  return a.i > static_cast<T>(b.width) && a.j > static_cast<T>(b.height);
}
template <typename T, typename TT>
__host__ __device__ bool operator>=(const Index2<T> &a, const Size2<TT> &b) {
  return a.i >= static_cast<T>(b.width) && a.j >= static_cast<T>(b.height);
}
/*****************************************************************************
**********************          INDEX2ITERATOR           *********************
******************************************************************************/
template <typename T> class Index2Iterator {
public:
  Index2Iterator() = default;
  __host__ __device__ Index2Iterator(Index2<T> lower, Index2<T> upper)
      : index_(lower), lower_(lower), upper_(upper) {}
  ///\brief Construct a new Index2Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  __host__ __device__ Index2Iterator(Index2<T> lower, Index2<T> upper,
                                     Index2<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  ///\return Index2Iterator&
  __host__ __device__ Index2Iterator &operator++() {
    index_.i++;
    if (index_.i >= upper_.i) {
      index_.i = lower_.i;
      index_.j++;
      if (index_.j >= upper_.j)
        index_ = upper_;
    }
    return *this;
  }
  ///\return const Index2<T>& current index coordinate
  __host__ __device__ const Index2<T> &operator*() const { return index_; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  __host__ __device__ bool operator==(const Index2Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  __host__ __device__ bool operator!=(const Index2Iterator<T> &other) const {
    return index_ != other.index_;
  }

private:
  Index2<T> index_, lower_, upper_;
};
/*****************************************************************************
***********************          INDEX2RANGE           ***********************
******************************************************************************/
/// Represents a closed-open range of indices [lower, upper),
/// Can be used in a for each loop
///\tparam T must be an integer type
template <typename T> class Index2Range {
public:
  ///\brief Construct a new Index2Range object
  ///\param upper_i **[in]** upper bound i
  ///\param upper_j **[in]** upper bound j
  __host__ __device__ Index2Range(T upper_i, T upper_j)
      : lower_(Index2<T>()), upper_(Index2<T>(upper_i, upper_j)) {}
  ///\brief Construct a new Index2Range object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  __host__ __device__ explicit Index2Range(Index2<T> lower,
                                           Index2<T> upper = Index2<T>())
      : lower_(lower), upper_(upper) {}
  __host__ __device__ explicit Index2Range(size2 upper)
      : lower_(Index2<T>()), upper_(Index2<T>(upper.width, upper.height)) {}
  ///\return Index2Iterator<T>
  __host__ __device__ Index2Iterator<T> begin() const {
    return Index2Iterator<T>(lower_, upper_, lower_);
  }
  ///\return Index2Iterator<T>
  __host__ __device__ Index2Iterator<T> end() const {
    return Index2Iterator<T>(lower_, upper_, upper_);
  }

private:
  Index2<T> lower_, upper_;
};
/*****************************************************************************
**************************          INDEX3           *************************
******************************************************************************/
/// Holds 3-dimensional index coordinates
///\tparam T must be an integer type
template <typename T> struct Index3 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index3 must hold an integer type!");
  ///\brief Construct a new Index2 object
  ///\param i **[in]** i coordinate value
  ///\param j **[in]** j coordinate value
  ///\param k **[in]** k coordinate value
  __host__ __device__ explicit Index3(T i = T(0), T j = T(0), T k = T(0))
      : i(i), j(j), k(k) {}
  __host__ __device__ T operator[](int _i) const { return (&i)[_i]; }
  __host__ __device__ T &operator[](int _i) { return (&i)[_i]; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if both coordinate values are equal
  __host__ __device__ bool operator==(const Index3<T> &other) const {
    return i == other.i && j == other.j && k == other.k;
  }
  /// \brief are different? operator
  ///\param other **[in]**
  ///\return bool true if any coordinate value is different
  __host__ __device__ bool operator!=(const Index3<T> &other) const {
    return i != other.i || j != other.j || k != other.k;
  }

  T i;
  T j;
  T k;
};
// ***********************************************************************
//                             BOOLEAN
// ***********************************************************************
template <typename T>
__host__ __device__ bool operator<=(const Index3<T> &a, const Index3<T> &b) {
  return a.i <= b.i && a.j <= b.j && a.k <= b.k;
}
template <typename T>
__host__ __device__ bool operator<(const Index3<T> &a, const Index3<T> &b) {
  return a.i < b.i && a.j < b.j && a.k < b.k;
}
template <typename T, typename TT>
__host__ __device__ bool operator<(const Index3<T> &a, const Size3<TT> &b) {
  return a.i < static_cast<T>(b.i) && a.j < static_cast<T>(b.j) &&
         a.k < static_cast<T>(b.k);
}
template <typename T, typename TT>
__host__ __device__ bool operator>(const Index3<T> &a, const Size3<TT> &b) {
  return a.i > static_cast<T>(b.i) && a.j > static_cast<T>(b.j) &&
         a.k > static_cast<T>(b.k);
}
template <typename T, typename TT>
__host__ __device__ bool operator>=(const Index3<T> &a, const Size3<TT> &b) {
  return a.i >= static_cast<T>(b.i) && a.j >= static_cast<T>(b.j) &&
         a.k >= static_cast<T>(b.k);
}
/*****************************************************************************
**********************          INDEX3ITERATOR           *********************
******************************************************************************/
template <typename T> class Index3Iterator {
public:
  ///\brief Construct a new Index3Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  __host__ __device__ Index3Iterator(Index3<T> lower, Index3<T> upper,
                                     Index3<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  ///\return Index3Iterator&
  __host__ __device__ Index3Iterator &operator++() {
    index_.i++;
    if (index_.i >= upper_.i) {
      index_.i = lower_.i;
      index_.j++;
      if (index_.j >= upper_.j) {
        index_.j = lower_.j;
        index_.k++;
        if (index_.k >= upper_.k)
          index_ = upper_;
      }
    }
    return *this;
  }
  ///\return const Index3<T>& current index coordinate
  __host__ __device__ const Index3<T> &operator*() const { return index_; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  __host__ __device__ bool operator==(const Index3Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  __host__ __device__ bool operator!=(const Index3Iterator<T> &other) const {
    return index_ != other.index_;
  }

private:
  Index3<T> index_, lower_, upper_;
};
/*****************************************************************************
***********************          INDEX3RANGE           ***********************
******************************************************************************/
/// Represents a closed-open range of indices [lower, upper),
/// Can be used in a for each loop
///\tparam T must be an integer type
template <typename T> class Index3Range {
public:
  ///\brief Construct a new Index3Range object
  ///\param upper_i **[in]** upper bound i
  ///\param upper_j **[in]** upper bound j
  ///\param upper_k **[in]** upper bound k
  __host__ __device__ Index3Range(T upper_i, T upper_j, T upper_k)
      : lower_(Index3<T>()), upper_(Index3<T>(upper_i, upper_j, upper_k)) {}
  ///\brief Construct a new Index3Range object
  ///\param upper **[in]** upper bound
  ///\param lower **[in]** lower bound
  __host__ __device__ explicit Index3Range(Index3<T> upper,
                                           Index3<T> lower = Index3<T>())
      : lower_(lower), upper_(upper) {}
  __host__ __device__ explicit Index3Range(size3 upper)
      : lower_(Index3<T>()),
        upper_(Index3<T>(upper.width, upper.height, upper.depth)) {}
  ///\return Index3Iterator<T>
  __host__ __device__ Index3Iterator<T> begin() const {
    return Index3Iterator<T>(lower_, upper_, lower_);
  }
  ///\return Index3Iterator<T>
  __host__ __device__ Index3Iterator<T> end() const {
    return Index3Iterator<T>(lower_, upper_, upper_);
  }

private:
  Index3<T> lower_, upper_;
};

using index2 = Index2<i32>;
using index2_8 = Index2<i8>;
using index2_16 = Index2<i16>;
using index2_32 = Index2<i32>;
using index2_64 = Index2<i64>;
using index3 = Index3<i32>;
using index3_8 = Index3<i8>;
using index3_16 = Index3<i16>;
using index3_32 = Index3<i32>;
using index3_64 = Index3<i64>;

} // namespace cuda

} // namespace hermes

#endif