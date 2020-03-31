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
/// \date 2020-02-25
///
/// \brief

#ifndef HERMES_BLAS_VECTOR_H
#define HERMES_BLAS_VECTOR_H

#include <hermes/common/reduce.h>
#include <hermes/storage/array.h>
#include <ponos/blas/vector.h>

namespace hermes {

namespace cuda {

// ***********************************************************************
//                           OPERATION KERNELS
// ***********************************************************************

/// Set of pre-built predicates to use with vector operations
class BlasVectorOperations {
public:
  template <typename T> struct sum {
    __host__ __device__ void operator()(const T &a, const T &b, T &c) {
      c = a + b;
    }
  };
  template <typename T> struct sub {
    __host__ __device__ void operator()(const T &a, const T &b, T &c) {
      c = a - b;
    }
  };
  template <typename T> struct mul {
    __host__ __device__ void operator()(const T &a, const T &b, T &c) {
      c = a * b;
    }
  };
  template <typename T> struct div {
    __host__ __device__ void operator()(const T &a, const T &b, T &c) {
      c = a / b;
    }
  };
};

/// Auxiliary class that encapsulates a c++ lambda function into device code
///\tparam T array data type
///\tparam F lambda function type following the signature:
/// (const T&, const T&, T&)
template <typename T, typename F> struct blas_vector {
  __host__ __device__ explicit blas_vector(const F &op) : operation(op) {}
  __host__ __device__ void operator()(const T &a, const T &b, T &c) {
    operation(a, b, c);
  }
  F operation;
};

template <typename T, typename F>
__global__ void __compute(Array1CAccessor<T> a, Array1CAccessor<T> b,
                          Array1Accessor<T> c, blas_vector<T, F> operation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (a.contains(i) && b.contains(i) && c.contains(i))
    operation(a[i], b[i], c[i]);
}
/// \tparam T
/// \tparam F
/// \param a **[in]**
/// \param b **[in]**
/// \param c **[out]**
/// \param operation **[in]**
template <typename T, typename F>
void compute(Array1CAccessor<T> a, Array1CAccessor<T> b, Array1Accessor<T> c,
             F operation) {
  blas_vector<T, F> vector(operation);
  ThreadArrayDistributionInfo td(a.size());
  __compute<T><<<td.gridSize, td.blockSize>>>(a, b, c, vector);
}

template <typename T, typename F>
__global__ void __compute(Array1CAccessor<T> a, T b, Array1Accessor<T> c,
                          blas_vector<T, F> operation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (a.contains(i) && c.contains(i))
    operation(a[i], b, c[i]);
}
/// \tparam T
/// \tparam F
/// \param a **[in]**
/// \param b **[in]**
/// \param c **[out]**
/// \param operation **[in]**
template <typename T, typename F>
void compute(Array1CAccessor<T> a, T b, Array1Accessor<T> c, F operation) {
  blas_vector<T, F> vector(operation);
  ThreadArrayDistributionInfo td(a.size());
  __compute<T><<<td.gridSize, td.blockSize>>>(a, b, c, vector);
}

// ***********************************************************************
//                           VECTOR
// ***********************************************************************

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
  Vector(u32 size, T value) {
    data_.resize(size);
    data_ = value;
  }
  /// \param other **[in]**
  Vector(const Vector<T> &other) { data_ = other.data_; }
  /// \param ponos_vector **[in]**
  Vector(const ponos::Vector<T> &ponos_vector) { data_ = ponos_vector.data(); }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  ///\param f **[in]**
  ///\return Vector<T>&
  Vector<T> &operator=(const Vector<T> &other) {
    data_ = other.data_;
    return *this;
  }
  ///\param f **[in]**
  ///\return Vector<T>&
  Vector<T> &operator=(T f) {
    data_ = f;
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator+=(const Vector<T> &v) {
    compute(data_.constAccessor(), v.data_.constAccessor(), data_.accessor(),
            BlasVectorOperations::sum<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator-=(const Vector<T> &v) {
    compute(data_.constAccessor(), v.data_.constAccessor(), data_.accessor(),
            BlasVectorOperations::sub<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator*=(const Vector<T> &v) {
    compute(data_.constAccessor(), v.data_.constAccessor(), data_.accessor(),
            BlasVectorOperations::mul<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator/=(const Vector<T> &v) {
    compute(data_.constAccessor(), v.data_.constAccessor(), data_.accessor(),
            BlasVectorOperations::div<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator+=(T v) {
    compute(data_.constAccessor(), v, data_.accessor(),
            BlasVectorOperations::sum<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator-=(T v) {
    compute(data_.constAccessor(), v, data_.accessor(),
            BlasVectorOperations::sub<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator*=(T v) {
    compute(data_.constAccessor(), v, data_.accessor(),
            BlasVectorOperations::mul<T>());
    return *this;
  }
  /// \param v **[in]**
  /// \return Vector<T>&
  Vector<T> &operator/=(T v) {
    compute(data_.constAccessor(), v, data_.accessor(),
            BlasVectorOperations::div<T>());
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// copies data to host side
  ///\return ponos::Vector<T> data in host side
  ponos::Vector<T> hostData() const {
    ponos::Vector<T> h_data = data_.hostData();
    return h_data;
  }
  /// \return u32
  u32 size() const { return data_.size(); }
  void resize(u32 new_size) { data_.resize(new_size); }
  const Array1<T> &data() const { return data_; }
  Array1<T> &data() { return data_; }

private:
  Array1<T> data_;
};

// ***********************************************************************
//                           ARITHMETIC
// ***********************************************************************
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b) {
  Vector<T> r(a.size());
  compute(a.data().constAccessor(), b.data().constAccessor(),
          r.data().accessor(), BlasVectorOperations::sum<T>());
  return r;
}
template <typename T>
Vector<T> operator-(const Vector<T> &a, const Vector<T> &b) {
  Vector<T> r(a.size());
  compute(a.data().constAccessor(), b.data().constAccessor(),
          r.data().accessor(), BlasVectorOperations::sub<T>());
  return r;
}
template <typename T> Vector<T> operator*(const Vector<T> &a, T f) {
  Vector<T> r(a.size());
  compute(a.data().constAccessor(), f, r.data().accessor(),
          BlasVectorOperations::mul<T>());
  return r;
}
template <typename T> Vector<T> operator/(const Vector<T> &a, T f) {
  Vector<T> r(a.size());
  compute(a.data().constAccessor(), 1 / f, r.data().accessor(),
          BlasVectorOperations::mul<T>());
  return r;
}
template <typename T> Vector<T> operator*(T f, const Vector<T> &v) {
  return v * f;
}
template <typename T> Vector<T> operator/(T f, const Vector<T> &v) {
  Vector<T> r(v.size(), f);
  compute(r.data().constAccessor(), v.data().constAccessor(),
          r.data().accessor(), BlasVectorOperations::div<T>());
  return r;
}
// ***********************************************************************
//                           BOOLEAN
// ***********************************************************************

template <typename T> bool operator==(const Vector<T> &a, const Vector<T> &b) {
  return reduce<T, u8, ReducePredicates::is_equal<T>>(
      a.data(), b.data(), ReducePredicates::is_equal<T>());
}

template <typename T> bool operator==(const Vector<T> &a, T f) {
  return reduce<T, u8, ReducePredicates::is_equal_to_value<T>>(
      a.data(), ReducePredicates::is_equal_to_value<T>(f));
}

} // namespace cuda

} // namespace hermes

#endif