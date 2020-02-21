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

#ifndef HERMES_GEOMETRY_CUDA_MATRIX_H
#define HERMES_GEOMETRY_CUDA_MATRIX_H

#include <cstring>
#include <cuda_runtime.h>
#include <hermes/geometry/vector.h>

namespace hermes {

namespace cuda {

template <typename T> class Matrix4x4 {
public:
  /// \param isIdentity [optional | def = true] initialize as an identity matrix
  __host__ __device__ explicit Matrix4x4(bool isIdentity = true);
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  __host__ __device__ Matrix4x4(std::initializer_list<T> values,
                                bool columnMajor = false);
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  __host__ __device__ explicit Matrix4x4(const T mat[16],
                                         bool columnMajor = false);
  /// \param mat matrix entries in [ROW][COLUMN] form
  __host__ __device__ explicit Matrix4x4(T mat[4][4]);
  /// \param m00 value of entry at row 0 column 0
  /// \param m01 value of entry at row 0 column 1
  /// \param m02 value of entry at row 0 column 2
  /// \param m03 value of entry at row 0 column 3
  /// \param m10 value of entry at row 1 column 0
  /// \param m11 value of entry at row 1 column 1
  /// \param m12 value of entry at row 1 column 2
  /// \param m13 value of entry at row 1 column 3
  /// \param m20 value of entry at row 2 column 0
  /// \param m21 value of entry at row 2 column 1
  /// \param m22 value of entry at row 2 column 2
  /// \param m23 value of entry at row 2 column 3
  /// \param m30 value of entry at row 3 column 0
  /// \param m31 value of entry at row 3 column 1
  /// \param m32 value of entry at row 3 column 2
  /// \param m33 value of entry at row 3 column 3
  __host__ __device__ Matrix4x4(T m00, T m01, T m02, T m03, T m10, T m11, T m12,
                                T m13, T m20, T m21, T m22, T m23, T m30, T m31,
                                T m32, T m33);
  __host__ __device__ void setIdentity();
  __host__ __device__ void row_major(T *a) const;
  __host__ __device__ void column_major(T *a) const;
  __host__ __device__ Matrix4x4 operator*(const Matrix4x4 &mat);
  __host__ __device__ bool operator==(const Matrix4x4 &_m) const;
  __host__ __device__ bool operator!=(const Matrix4x4 &_m) const;
  __host__ __device__ bool isIdentity();
  __host__ __device__ static Matrix4x4 mul(const Matrix4x4 &m1,
                                           const Matrix4x4 &m2);
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Matrix4x4<TT> &m);
  T m[4][4];
};

template <typename T>
__host__ __device__ Matrix4x4<T> transpose(const Matrix4x4<T> &m);
template <typename T>
__host__ __device__ Matrix4x4<T> inverse(const Matrix4x4<T> &m);

template <typename T> class Matrix3x3 {
public:
  __host__ __device__ Matrix3x3();
  __host__ __device__ Matrix3x3(vec3 a, vec3 b, vec3 c);
  __host__ __device__ Matrix3x3(T m00, T m01, T m02, T m10, T m11, T m12, T m20,
                                T m21, T m22);
  __host__ __device__ void setIdentity();
  __host__ __device__ Vector3<T> operator*(const Vector3<T> &v) const;
  __host__ __device__ Matrix3x3 operator*(const Matrix3x3 &mat);
  __host__ __device__ Matrix3x3 operator*(const T &f) const;
  __host__ __device__ static Matrix3x3 mul(const Matrix3x3 &m1,
                                           const Matrix3x3 &m2);
  __host__ __device__ T determinant();
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Matrix3x3<TT> &m);
  T m[3][3];
};

template <typename T>
__host__ __device__ inline Matrix3x3<T> operator*(T f, const Matrix3x3<T> &m) {
  return m * f;
}
template <typename T>
__host__ __device__ Matrix3x3<T> transpose(const Matrix3x3<T> &m);
template <typename T>
__host__ __device__ Matrix3x3<T> inverse(const Matrix3x3<T> &m);
template <typename T>
__host__ __device__ Matrix3x3<T> star(const Vector3<T> &a);

typedef Matrix4x4<float> mat4;
typedef Matrix3x3<float> mat3;

#include "cuda_matrix.inl"

} // namespace cuda

} // namespace hermes

#endif // HERMES_GEOMETRY_CUDA_MATRIX_H