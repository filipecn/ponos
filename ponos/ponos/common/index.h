/// Copyright (c) 2019, FilipeCN.
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
/// \file index.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2019-08-16
///
/// \brief Representation of index coordinates

#ifndef PONOS_COMMON_INDEX_H
#define PONOS_COMMON_INDEX_H

#include <algorithm>
#include <iostream>
#include <ponos/common/defs.h>
#include <ponos/common/size.h>
#include <cstdlib> // abs

namespace ponos {

/*****************************************************************************
**************************          INDEX2           *************************
******************************************************************************/
/// Holds 2-dimensional index coordinates
///
/// - Usually the field ``i`` is related to the **x** axis in cartesian
/// coordinates, and the field ``j`` is related to the **y** axis.
///\tparam T index type
/// \verbatim embed:rst:leading-slashes
///    .. warning::
///       Index type must be a signed integer type.
/// \endverbatim
template<typename T> struct Index2 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index2 must hold an integer type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Index2() = default;
  /// Constructor
  /// \param v **[in]** value assigned to both ``i`` and ``j``
  explicit Index2(T v) : i(v), j(v) {}
  ///\brief Constructor
  ///\param i **[in]** coordinate value for ``i``
  ///\param j **[in]** coordinate value for ``j``
  explicit Index2(T i, T j) : i(i), j(j) {}
  /// \brief Constructor from a Size2 object
  /// - ``i`` receives ``size.with`` and ``j`` receives ``size.height``
  /// \tparam S size type
  /// \param size **[in]**
  template<typename S>
  explicit Index2(const Size2<S> &size) : i(size.width), j(size.height) {}
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does not check if ``d`` is out of bounds.
  /// \endverbatim
  /// \param d **[in]** dimension number (``0`` for ``i`` and ``1`` for ``j``)
  /// \return T coordinate value at dimension ``d``
  T operator[](int d) const { return (&i)[d]; }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does not check if ``d`` is out of bounds.
  /// \endverbatim
  /// \param d **[in]** dimension number (``0`` for ``i`` and ``1`` for ``j``)
  /// \return T reference to coordinate value at dimension ``d``
  T &operator[](int d) { return (&i)[d]; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  /// Generates an index with incremented values
  /// \param _i **[in]** value incremented to ``i``
  /// \param _j **[in]** value incremented to ``j``
  /// \return Index2<T> resulting index coordinates
  Index2<T> plus(T _i, T _j) const { return Index2<T>(i + _i, j + _j); }
  /// Generates a copy with ``i`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i-d``, ``j``)
  Index2<T> left(T d = T(1)) const { return Index2<T>(i - d, j); }
  /// Generates a copy with ``i`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i+d``, ``j``)
  Index2<T> right(T d = T(1)) const { return Index2<T>(i + d, j); }
  /// Generates a copy with ``j`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i``, ``j-d``)
  Index2<T> down(T d = T(1)) const { return Index2<T>(i, j - d); }
  /// Generates a copy with ``j`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i``, ``j+d``)
  Index2<T> up(T d = T(1)) const { return Index2<T>(i, j + d); }
  /// Clamps to the inclusive range ``[0, size]``
  /// \param s **[in]** upper bound
  void clampTo(const size2 &s) {
    i = std::max(0, std::min(i, static_cast<T>(s.width)));
    j = std::max(0, std::min(j, static_cast<T>(s.height)));
  }
  // ***********************************************************************
  //                            FIELDS
  // ***********************************************************************
  /// 0-th coordinate value
  T i = T(0);
  /// 1-th coordinate value
  T j = T(0);
};
// ***********************************************************************
//                             ARITHMETIC
// ***********************************************************************
template<typename T>
Index2<T> operator+(const Index2<T> &a, const Index2<T> &b) {
  return Index2<T>(a.i + b.i, a.j + b.j);
}
template<typename T>
Index2<T> operator-(const Index2<T> &a, const Index2<T> &b) {
  return Index2<T>(a.i - b.i, a.j - b.j);
}
template<typename T, typename U>
Index2<T> operator+(const Index2<T> &a, const Size2<U> &b) {
  return Index2<T>(a.i + b.width, a.j + b.height);
}
template<typename U, typename T>
Index2<T> operator+(const Size2<U> &b, const Index2<T> &a) {
  return Index2<T>(a.i + b.width, a.j + b.height);
}
template<typename T, typename U>
Index2<T> operator-(const Index2<T> &a, const Size2<U> &b) {
  return Index2<T>(a.i - b.width, a.j - b.height);
}
template<typename U, typename T>
Index2<T> operator-(const Size2<U> &b, const Index2<T> &a) {
  return Index2<T>(b.width - a.i, b.height - a.j);
}
/// \brief Computes the manhattan distance between two indices
/// \tparam T
/// \param a **[in]**
/// \param b **[in]**
/// \return T
template<typename T> T distance(const Index2<T> &a, const Index2<T> &b) {
  return std::abs(a.i - b.i) + std::abs(a.j - b.j);
}
// ***********************************************************************
//                             BOOLEAN
// ***********************************************************************
template<typename T> bool operator<=(const Index2<T> &a, const Index2<T> &b) {
  return a.i <= b.i && a.j <= b.j;
}
///\brief are equal? operator
///\param other **[in]**
///\return bool true if both coordinate values are equal
template<typename T> bool operator==(const Index2<T> &a, const Index2<T> &b) {
  return a.i == b.i && a.j == b.j;
}
/// \brief are different? operator
///\param other **[in]**
///\return bool true if any coordinate value is different
template<typename T> bool operator!=(const Index2<T> &a, const Index2<T> &b) {
  return a.i != b.i || a.j != b.j;
}

/*****************************************************************************
**********************          INDEX2ITERATOR           *********************
******************************************************************************/
template<typename T> class Index2Iterator {
public:
  Index2Iterator() = default;
  Index2Iterator(Index2<T> lower, Index2<T> upper)
      : index_(lower), lower_(lower), upper_(upper) {}
  ///\brief Construct a new Index2Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  Index2Iterator(Index2<T> lower, Index2<T> upper, Index2<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  ///\return Index2Iterator&
  Index2Iterator &operator++() {
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
  const Index2<T> &operator*() const { return index_; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  bool operator==(const Index2Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  bool operator!=(const Index2Iterator<T> &other) const {
    return index_ != other.index_;
  }

private:
  Index2<T> index_, lower_, upper_;
};
/*****************************************************************************
***********************          INDEX2RANGE           ***********************
******************************************************************************/
/// Represents a closed-open range of indices ``[lower, upper)``
///
/// Can be used in a for each loop that iterates over all indices in the range:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       ponos::size2 size(10,10);
///       for(auto ij : ponos::Index2Range<int>(size)) {
///         *ij; // index coordinates
///       }
/// \endverbatim
///\tparam T must be an integer type
template<typename T> class Index2Range {
public:
  ///\brief Constructs an index range ``[0, {upper_i,upper_j})``
  ///\param upper_i **[in]** upper bound i
  ///\param upper_j **[in]** upper bound j
  Index2Range(T upper_i, T upper_j)
      : lower_(Index2<T>()), upper_(Index2<T>(upper_i, upper_j)) {}
  ///\brief Constructs an index range ``[lower, upper)``
  ///\param lower **[in]** lower bound
  ///\param upper **[in | default = Index2<T>()]** upper bound
  explicit Index2Range(Index2<T> lower, Index2<T> upper = Index2<T>())
      : lower_(lower), upper_(upper) {}
  /// \brief Constructs an index range ``[0, upper)``
  /// \param upper **[in]** upper bound
  explicit Index2Range(size2 upper)
      : lower_(Index2<T>()), upper_(Index2<T>(upper.width, upper.height)) {}
  Index2Iterator<T> begin() const {
    return Index2Iterator<T>(lower_, upper_, lower_);
  }
  Index2Iterator<T> end() const {
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
template<typename T> struct Index3 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index3 must hold an integer type!");
  Index3() : i(0), j(0), k(0) {}
  explicit Index3(T v) : i(v), j(v), k(v) {}
  ///\brief Construct a new Index2 object
  ///\param i **[in]** i coordinate value
  ///\param j **[in]** j coordinate value
  ///\param k **[in]** k coordinate value
  explicit Index3(T i, T j, T k) : i(i), j(j), k(k) {}
  T operator[](int _i) const { return (&i)[_i]; }
  T &operator[](int _i) { return (&i)[_i]; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if both coordinate values are equal
  bool operator==(const Index3<T> &other) const {
    return i == other.i && j == other.j && k == other.k;
  }
  /// \brief are different? operator
  ///\param other **[in]**
  ///\return bool true if any coordinate value is different
  bool operator!=(const Index3<T> &other) const {
    return i != other.i || j != other.j || k != other.k;
  }

  T i{0};
  T j{0};
  T k{0};
};
// ***********************************************************************
//                             ARITHMETIC
// ***********************************************************************
template<typename T>
Index3<T> operator+(const Index3<T> &a, const Index3<T> &b) {
  return Index3<T>(a.i + b.i, a.j + b.j, a.k + b.k);
}
template<typename T>
Index3<T> operator-(const Index3<T> &a, const Index3<T> &b) {
  return Index3<T>(a.i - b.i, a.j - b.j, a.k - b.k);
}
template<typename T, typename U>
Index3<T> operator+(const Index3<T> &a, const Size3<U> &b) {
  return Index3<T>(a.i + b.width, a.j + b.height, a.k + b.depth);
}
template<typename U, typename T>
Index3<T> operator+(const Size3<U> &b, const Index3<T> &a) {
  return Index3<T>(a.i + b.width, a.j + b.height, a.k + b.depth);
}
template<typename T, typename U>
Index3<T> operator-(const Index3<T> &a, const Size3<U> &b) {
  return Index3<T>(a.i - b.width, a.j - b.height, a.k - b.depth);
}
template<typename U, typename T>
Index3<T> operator-(const Size3<U> &b, const Index3<T> &a) {
  return Index3<T>(b.width - a.i, b.height - a.j, b.depth - a.k);
}
// ***********************************************************************
//                             BOOLEAN
// ***********************************************************************
template<typename T> bool operator<=(const Index3<T> &a, const Index3<T> &b) {
  return a.i <= b.i && a.j <= b.j && a.k <= b.k;
}
template<typename T> bool operator<(const Index3<T> &a, const Index3<T> &b) {
  return a.i < b.i && a.j < b.j && a.k < b.k;
}
template<typename T, typename TT>
bool operator<(const Index3<T> &a, const Size3<TT> &b) {
  return a.i < static_cast<T>(b.i) && a.j < static_cast<T>(b.j) &&
      a.k < static_cast<T>(b.k);
}
template<typename T, typename TT>
bool operator>(const Index3<T> &a, const Size3<TT> &b) {
  return a.i > static_cast<T>(b.i) && a.j > static_cast<T>(b.j) &&
      a.k > static_cast<T>(b.k);
}
template<typename T, typename TT>
bool operator>=(const Index3<T> &a, const Size3<TT> &b) {
  return a.i >= static_cast<T>(b.i) && a.j >= static_cast<T>(b.j) &&
      a.k >= static_cast<T>(b.k);
}
/*****************************************************************************
**********************          INDEX3ITERATOR           *********************
******************************************************************************/
template<typename T> class Index3Iterator {
public:
  ///\brief Construct a new Index3Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  Index3Iterator(Index3<T> lower, Index3<T> upper, Index3<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  ///\return Index3Iterator&
  Index3Iterator &operator++() {
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
  const Index3<T> &operator*() const { return index_; }
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  bool operator==(const Index3Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  bool operator!=(const Index3Iterator<T> &other) const {
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
template<typename T> class Index3Range {
public:
  ///\brief Construct a new Index3Range object
  ///\param upper_i **[in]** upper bound i
  ///\param upper_j **[in]** upper bound j
  ///\param upper_k **[in]** upper bound k
  Index3Range(T upper_i, T upper_j, T upper_k)
      : lower_(Index3<T>()), upper_(Index3<T>(upper_i, upper_j, upper_k)) {}
  ///\brief Construct a new Index3Range object
  ///\param upper **[in]** upper bound
  ///\param lower **[in]** lower bound
  explicit Index3Range(Index3<T> upper)
      : upper_(upper) {}
  explicit Index3Range(size3 upper)
      : lower_(Index3<T>()),
        upper_(Index3<T>(upper.width, upper.height, upper.depth)) {}
  explicit Index3Range(Index3<T> lower, Index3<T> upper) : lower_(lower), upper_(upper) {}
  ///\return Index3Iterator<T>
  Index3Iterator<T> begin() const {
    return Index3Iterator<T>(lower_, upper_, lower_);
  }
  ///\return Index3Iterator<T>
  Index3Iterator<T> end() const {
    return Index3Iterator<T>(lower_, upper_, upper_);
  }
  [[nodiscard]] size3 size() const {
    return size3(std::abs(upper_[0] - lower_[0]),
                 std::abs(upper_[1] - lower_[1]),
                 std::abs(upper_[2] - lower_[2]));
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

template<typename T>
std::ostream &operator<<(std::ostream &o, const Index2<T> &ij) {
  o << "Index[" << ij.i << ", " << ij.j << "]";
  return o;
}
template<typename T>
std::ostream &operator<<(std::ostream &o, const Index3<T> &ijk) {
  o << "Index[" << ijk.i << ", " << ijk.j << ", " << ijk.k << "]";
  return o;
}

} // namespace ponos

#endif
