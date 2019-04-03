/*
 * Copyright (c) 2017 FilipeCN
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

#ifndef PONOS_GEOMETRY_MATRIX_H
#define PONOS_GEOMETRY_MATRIX_H

#include <ponos/geometry/vector.h>
#include <vector>

namespace ponos {

template <typename T = float> class DenseMatrix {
public:
  DenseMatrix() : N(0), M(0), data(nullptr) {}
  DenseMatrix(size_t n) : N(n), M(n), data(nullptr) { set(n, n); }
  DenseMatrix(size_t n, size_t m) : N(n), M(m), data(nullptr) { set(n, m); }
  DenseMatrix(const DenseMatrix<T> &A) : DenseMatrix() {
    set(A.N, A.M);
    // TODO make this copy faster
    for (size_t r = 0; r < A.N; r++)
      for (size_t c = 0; c < A.M; c++)
        data[r][c] = A.data[r][c];
  }
  virtual ~DenseMatrix() {
    if (data) {
      for (size_t i = 0; i < N; i++)
        delete[] data[i];
      delete[] data;
    }
  }

  T &operator()(size_t i, size_t j) { return data[i][j]; }
  T operator()(size_t i, size_t j) const { return data[i][j]; }
  DenseMatrix<T> &operator=(DenseMatrix<T> A) {
    set(A.N, A.M);
    for (size_t r = 0; r < A.N; r++)
      std::copy(&A.data[r][0], &A.data[r][0] + A.M, &data[r][0]);
    return *this;
  }
  DenseMatrix<T> operator*(const DenseMatrix<T> &A) const {
    FATAL_ASSERT(M == A.N && N == A.M);
    DenseMatrix<T> R(N, A.M);
    for (size_t r = 0; r < R.N; r++)
      for (size_t c = 0; c < R.M; c++) {
        R(r, c) = 0;
        for (size_t i = 0; i < N; i++)
          R(r, c) += data[r][i] * A.data[c][i];
      }
    return R;
  }
  void set(size_t rows, size_t columns) {
    N = rows, M = columns;
    if (data) {
      for (size_t i = 0; i < N; i++)
        delete[] data[i];
      delete[] data;
    }
    data = new T *[N];
    for (size_t i = 0; i < N; i++) {
      data[i] = new T[M];
      memset(data[i], 0, sizeof(T) * M);
    }
  }

  size_t rowsNumber() const { return N; }
  size_t columnsNumber() const { return M; }

  friend std::ostream &operator<<(std::ostream &os, const DenseMatrix<T> &m) {
    for (size_t i = 0; i < m.N; i++) {
      for (size_t j = 0; j < m.M; j++)
        os << m(i, j) << " ";
      os << std::endl;
    }
    return os;
  }

private:
  size_t N, M;
  T **data;
};

template <typename T>
inline std::vector<T> operator*(const std::vector<T> &v, DenseMatrix<T> &m) {
  ASSERT_FATAL(v.size() == m.rowsNumber());
  std::vector<T> r;
  for (size_t c = 0; c < m.columnsNumber(); c++) {
    T sum;
    for (size_t i = 0; i < v.size(); i++)
      sum = sum + v[i] * m(i, c);
    r.emplace_back(sum);
  }
  return r;
}

template <typename T = float, size_t N = 5, size_t M = 5> class Matrix {
public:
  Matrix() {}
  Matrix(std::initializer_list<T> vs, bool cm = false) {
    size_t l = 0, c = 0;
    for (auto v = vs.begin(); v != vs.end(); v++) {
      m[l][c] = *v;
      if (!cm) {
        c++;
        if (c >= M)
          c = 0, l++;
      } else {
        l++;
        if (l >= N)
          l = 0, c++;
      }
    }
  }
  Vector<T, N> operator*(const Vector<T, M> &v) {
    Vector<T, N> x;
    for (size_t i = 0; i < N; i++) {
      x[i] = static_cast<T>(0);
      for (size_t j = 0; j < M; j++)
        x[i] += v[j] * m[i][j];
    }
    return x;
  }
  template <size_t P> Matrix<T, N, P> operator*(const Matrix<T, M, P> &b) {
    Matrix<T, N, P> c;
    for (size_t i = 0; i < N; i++)
      for (size_t j = 0; j < P; j++) {
        c.m[i][j] = static_cast<T>(0);
        for (size_t k = 0; k < M; k++)
          c.m[i][j] += m[i][k] * b.m[k][j];
      }
    return c;
  }

  T m[N][M];
};

template <typename T, size_t N>
Matrix<T, N, N> inverse(const Matrix<T, N, N> &m) {
  Vector<T, 2 * N> em[N];
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++) {
      em[i][j] = m.m[i][j];
      if (i == j)
        em[i][j + N] = 1.f;
      else
        em[i][j + N] = 0.f;
    }
  // Gauss-Jordan Elimination
  for (size_t k = 0; k < N; k++) {
    em[k] /= em[k][k];
    for (size_t i = 0; i < N; i++) {
      if (i != k)
        em[i] -= em[k] * em[i][k];
    }
  }
  Matrix<T, N, N> r;
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      r.m[i][j] = em[i][j + N];
  return r;
}

/// 4x4 Matrix representation
/// Access: m[ROW][COLUMN]
template <typename T> class Matrix4x4 {
public:
  /// \param isIdentity [optional | def = true] initialize as an identity matrix
  explicit Matrix4x4(bool isIdentity = true);
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  Matrix4x4(std::initializer_list<T> values, bool columnMajor = false);
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  explicit Matrix4x4(const T mat[16], bool columnMajor = false);
  /// \param mat matrix entries in [ROW][COLUMN] form
  explicit Matrix4x4(T mat[4][4]);
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
  Matrix4x4(T m00, T m01, T m02, T m03, T m10, T m11, T m12, T m13, T m20,
            T m21, T m22, T m23, T m30, T m31, T m32, T m33);
  void setIdentity();
  void row_major(T *a) const;
  void column_major(T *a) const;
  Matrix4x4 operator*(const Matrix4x4 &mat);
  Vector4<T> operator*(const Vector4<T> &v) const;
  bool operator==(const Matrix4x4 &_m) const;
  bool operator!=(const Matrix4x4 &_m) const;
  bool isIdentity();
  static Matrix4x4 mul(const Matrix4x4 &m1, const Matrix4x4 &m2);
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Matrix4x4<TT> &m);
  T m[4][4];
};

typedef Matrix4x4<real_t> mat4;

template <typename T>
Matrix4x4<T> rowReduce(const Matrix4x4<T> &p, const Matrix4x4<T> &q);
template <typename T> Matrix4x4<T> transpose(const Matrix4x4<T> &m);
template <typename T> Matrix4x4<T> inverse(const Matrix4x4<T> &m);
template <typename T>
void decompose(const Matrix4x4<T> &m, Matrix4x4<T> &r, Matrix4x4<T> &s);

template <typename T> class Matrix3x3 {
public:
  Matrix3x3();
  Matrix3x3(vec3 a, vec3 b, vec3 c);
  Matrix3x3(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22);
  void setIdentity();
  Vector3<T> operator*(const Vector3<T> &v) const;
  Matrix3x3 operator*(const Matrix3x3 &mat);
  Matrix3x3 operator*(const T &f) const;
  static Matrix3x3 mul(const Matrix3x3 &m1, const Matrix3x3 &m2);
  T determinant();
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Matrix3x3<TT> &m);
  T m[3][3];
};

template <typename T>
inline Matrix3x3<T> operator*(T f, const Matrix3x3<T> &m) {
  return m * f;
}

typedef Matrix3x3<real_t> mat3;

template <typename T> Matrix3x3<T> transpose(const Matrix3x3<T> &m);
template <typename T> Matrix3x3<T> inverse(const Matrix3x3<T> &m);
template <typename T> Matrix3x3<T> star(const Vector3<T> &a);

template <typename T> class Matrix2x2 {
public:
  Matrix2x2();
  Matrix2x2(T m00, T m01, T m10, T m11);
  void setIdentity();
  T determinant();
  Vector2<T> operator*(const Vector2<T> &v) const;
  Matrix2x2 operator*(const Matrix2x2 &mat);
  Matrix2x2 operator*(const T &f) const;
  static Matrix2x2 mul(const Matrix2x2 &m1, const Matrix2x2 &m2);
  template <typename TT>
  friend std::ostream &operator<<(std::ostream &os, const Matrix2x2<TT> &m);
  T m[2][2];
};

typedef Matrix2x2<real_t> mat2;

template <typename T>
inline Matrix2x2<T> operator*(T f, const Matrix2x2<T> &m) {
  return m * f;
}

template <typename T> Matrix2x2<T> transpose(const Matrix2x2<T> &m);
template <typename T> Matrix2x2<T> inverse(const Matrix2x2<T> &m);

#include "matrix.inl"

} // namespace ponos

#endif
