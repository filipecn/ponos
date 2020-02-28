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
/// \file vector.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-02-24
///
/// \brief

#ifndef PONOS_BLAS_VECTOR_H
#define PONOS_BLAS_VECTOR_H

#include <ponos/common/defs.h>
#include <vector>

namespace ponos {

/// Represents a numerical vector that can be used in BLAS operations
/// \tparam T vector data type
template <typename T> class Vector {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Vector must hold an float type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Vector() = default;
  /// \param size **[in]**
  Vector(u32 size) { data_.resize(size); }
  /// \param size **[in]**
  /// \param value **[in]**
  Vector(u32 size, T value) { data_.resize(size, value); }
  /// \param other **[in]**
  Vector(const Vector<T> &other) { data_ = other.data_; }
  /// \param v **[in]**
  Vector(const std::vector<T> &v) { data_ = v; }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  ///\param v **[in]**
  ///\return Vector<T>&
  Vector<T> &operator=(const std::vector<T> &v) {
    data_ = v;
    return *this;
  }
  ///\param other **[in]**
  ///\return Vector<T>&
  Vector<T> &operator=(const Vector<T> &other) {
    data_ = other.data_;
    return *this;
  }
  ///\param f **[in]**
  ///\return Vector<T>&
  Vector<T> &operator=(T f) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] = f;
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator+=(const Vector<T> &v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] += v.data_[i];
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator-=(const Vector<T> &v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] -= v.data_[i];
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator*=(const Vector<T> &v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] *= v.data_[i];
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator/=(const Vector<T> &v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] /= v.data_[i];
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator+=(T v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] += v;
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator-=(T v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] -= v;
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator*=(T v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] *= v;
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator/=(T v) {
    for (size_t i = 0; i < data_.size(); ++i)
      data_[i] /= v;
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \param i **[in]**
  /// \return T
  T operator[](u32 i) const { return data_[i]; }
  /// \param i **[in]**
  /// \return T&
  T &operator[](u32 i) { return data_[i]; }
  const std::vector<T> &data() const { return data_; }
  std::vector<T> &data() { return data_; }
  /// \return u32
  u32 size() const { return data_.size(); }

private:
  std::vector<T> data_;
};

// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b) {
  Vector<T> r(a.size());
  for (u32 i = 0; i < a.size(); ++i)
    r[i] = a[i] + b[i];
  return r;
}
template <typename T>
Vector<T> operator-(const Vector<T> &a, const Vector<T> &b) {
  Vector<T> r(a.size());
  for (u32 i = 0; i < a.size(); ++i)
    r[i] = a[i] - b[i];
  return r;
}
template <typename T> Vector<T> operator*(const Vector<T> &a, T f) {
  Vector<T> r(a.size());
  for (u32 i = 0; i < a.size(); ++i)
    r[i] = a[i] * f;
  return r;
}
template <typename T> Vector<T> operator/(const Vector<T> &a, T f) {
  T inv = 1 / f;
  Vector<T> r(a.size());
  for (u32 i = 0; i < a.size(); ++i)
    r[i] = a[i] * inv;
  return r;
}
template <typename T> Vector<T> operator*(T f, const Vector<T> &v) {
  return v * f;
}
template <typename T> Vector<T> operator/(T f, const Vector<T> &v) {
  Vector<T> r(v.size());
  for (u32 i = 0; i < v.size(); ++i)
    r[i] = f / v[i];
  return r;
}
// ***********************************************************************
//                           BOOLEAN
// ***********************************************************************
template <typename T> bool operator==(const Vector<T> &a, const Vector<T> &b) {
  for (u32 i = 0; i < a.size(); ++i)
    if (!Check::is_equal(a[i], b[i]))
      return false;
  return true;
}
template <typename T> bool operator==(const Vector<T> &a, T f) {
  for (u32 i = 0; i < a.size(); ++i)
    if (!Check::is_equal(a[i], f))
      return false;
  return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &ost, const Vector<T> &v) {
  ost << "Vector<T>\n";
  for (u32 i = 0; i < v.size(); ++i)
    ost << v[i] << std::endl;
  return ost;
}

} // namespace ponos

#endif