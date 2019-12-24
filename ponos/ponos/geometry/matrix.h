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
/*
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
}*/

/// 4x4 Matrix representation
/// Access: m[ROW][COLUMN]
template <typename T> class Matrix4x4 {
public:
  /// \param isIdentity [optional | def = true] initialize as an identity matrix
  explicit Matrix4x4(bool isIdentity = true) {
    memset(m, 0, sizeof(m));
    if (isIdentity)
      for (int i = 0; i < 4; i++)
        m[i][i] = 1.f;
  }
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  Matrix4x4(std::initializer_list<T> values, bool columnMajor = false) {
    size_t l = 0, c = 0;
    for (auto v : values) {
      m[l][c] = v;
      if (columnMajor) {
        l++;
        if (l >= 4)
          l = 0, c++;
      } else {
        c++;
        if (c >= 4)
          c = 0, l++;
      }
    }
  }
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  explicit Matrix4x4(const T mat[16], bool columnMajor = false) {
    size_t k = 0;
    if (columnMajor)
      for (int c = 0; c < 4; c++)
        for (int l = 0; l < 4; l++)
          m[l][c] = mat[k++];
    else
      for (int l = 0; l < 4; l++)
        for (int c = 0; c < 4; c++)
          m[l][c] = mat[k++];
  }

  /// \param mat matrix entries in [ROW][COLUMN] form
  explicit Matrix4x4(T mat[4][4]) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        m[i][j] = mat[i][j];
  }
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
            T m21, T m22, T m23, T m30, T m31, T m32, T m33) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
  }

  void setIdentity() {
    memset(m, 0, sizeof(m));
    for (int i = 0; i < 4; i++)
      m[i][i] = 1.f;
  }
  void row_major(T *a) const {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        a[k++] = m[i][j];
  }
  void column_major(T *a) const {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        a[k++] = m[j][i];
  }
  bool isIdentity() {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (i != j && !Check::is_equal(m[i][j], 0.f))
          return false;
        else if (i == j && !Check::is_equal(m[i][j], 1.f))
          return false;
    return true;
  }
  T m[4][4];
};

template <typename T>
Matrix4x4<T> operator*(const Matrix4x4<T> &A, const Matrix4x4<T> &B) {
  Matrix4x4<T> r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      r.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] +
                  A.m[i][2] * B.m[2][j] + A.m[i][3] * B.m[3][j];
  return r;
}
template <typename T>
Vector4<T> operator*(const Matrix4x4<T> &A, const Vector4<T> &v) {
  Vector4<T> r;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      r[i] += A.m[i][j] * v[j];
  return r;
}
template <typename T>
bool operator==(const Matrix4x4<T> &A, const Matrix4x4<T> &B) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (!Check::is_equal(A.m[i][j], B.m[i][j]))
        return false;
  return true;
}
template <typename T>
bool operator!=(const Matrix4x4<T> &A, const Matrix4x4<T> &B) {
  return !(A == B);
}

typedef Matrix4x4<real_t> mat4;

template <typename T>
Matrix4x4<T> rowReduce(const Matrix4x4<T> &p, const Matrix4x4<T> &q) {
  Matrix4x4<T> l = p, r = q;
  // TODO implement with gauss jordan elimination
  return r;
}
template <typename T> Matrix4x4<T> transpose(const Matrix4x4<T> &m) {
  return Matrix4x4<T>(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                      m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                      m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                      m.m[3][3]);
}
// function extracted from MESA implementation of the GLU library
template <typename T> bool gluInvertMatrix(const T m[16], T invOut[16]) {
  T inv[16], det;
  int i;

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
           m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
           m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
           m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
           m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
           m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
           m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
           m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
           m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0f / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}
template <typename T> Matrix4x4<T> inverse(const Matrix4x4<T> &m) {
  Matrix4x4<T> r;
  T mm[16], inv[16];
  m.row_major(mm);
  if (gluInvertMatrix(mm, inv)) {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r.m[i][j] = inv[k++];
    return r;
  }

  T det = m.m[0][0] * m.m[1][1] * m.m[2][2] * m.m[3][3] +
          m.m[1][2] * m.m[2][3] * m.m[3][1] * m.m[1][3] +
          m.m[2][1] * m.m[3][2] * m.m[1][1] * m.m[2][3] +
          m.m[3][2] * m.m[1][2] * m.m[2][1] * m.m[3][3] +
          m.m[1][3] * m.m[2][2] * m.m[3][1] * m.m[0][1] +
          m.m[0][1] * m.m[2][3] * m.m[3][2] * m.m[0][2] +
          m.m[2][1] * m.m[3][3] * m.m[0][3] * m.m[2][2] +
          m.m[3][1] * m.m[0][1] * m.m[2][2] * m.m[3][3] +
          m.m[0][2] * m.m[2][3] * m.m[3][1] * m.m[0][3] +
          m.m[2][1] * m.m[3][2] * m.m[0][2] * m.m[0][1] +
          m.m[1][2] * m.m[3][3] * m.m[0][2] * m.m[1][3] +
          m.m[3][1] * m.m[0][3] * m.m[1][1] * m.m[3][2] -
          m.m[0][1] * m.m[1][3] * m.m[3][2] * m.m[0][2] -
          m.m[1][1] * m.m[3][3] * m.m[0][3] * m.m[1][2] -
          m.m[3][1] * m.m[0][3] * m.m[0][1] * m.m[1][3] -
          m.m[2][2] * m.m[0][2] * m.m[1][1] * m.m[2][3] -
          m.m[0][3] * m.m[1][2] * m.m[2][1] * m.m[0][1] -
          m.m[1][2] * m.m[2][3] * m.m[0][2] * m.m[1][3] -
          m.m[2][1] * m.m[0][3] * m.m[1][1] * m.m[2][2] -
          m.m[1][0] * m.m[1][0] * m.m[2][3] * m.m[3][2] -
          m.m[1][2] * m.m[2][0] * m.m[3][3] * m.m[1][3] -
          m.m[2][2] * m.m[3][0] * m.m[1][0] * m.m[2][2] -
          m.m[3][3] * m.m[1][2] * m.m[2][3] * m.m[3][0] -
          m.m[1][3] * m.m[2][0] * m.m[3][2] * m.m[1][1];
  if (fabs(det) < 1e-8)
    return r;

  r.m[0][0] =
      (m.m[1][1] * m.m[2][2] * m.m[3][3] + m.m[1][2] * m.m[2][3] * m.m[3][1] +
       m.m[1][3] * m.m[2][1] * m.m[3][2] - m.m[1][1] * m.m[2][3] * m.m[3][2] -
       m.m[1][2] * m.m[2][1] * m.m[3][3] - m.m[1][3] * m.m[2][2] * m.m[3][1]) /
      det;
  r.m[0][1] =
      (m.m[0][1] * m.m[2][3] * m.m[3][2] + m.m[0][2] * m.m[2][1] * m.m[3][3] +
       m.m[0][3] * m.m[2][2] * m.m[3][1] - m.m[0][1] * m.m[2][2] * m.m[3][3] -
       m.m[0][2] * m.m[2][3] * m.m[3][1] - m.m[0][3] * m.m[2][1] * m.m[3][2]) /
      det;
  r.m[0][2] =
      (m.m[0][1] * m.m[1][2] * m.m[3][3] + m.m[0][2] * m.m[1][3] * m.m[3][1] +
       m.m[0][3] * m.m[1][1] * m.m[3][2] - m.m[0][1] * m.m[1][3] * m.m[3][2] -
       m.m[0][2] * m.m[1][1] * m.m[3][3] - m.m[0][3] * m.m[1][2] * m.m[3][1]) /
      det;
  r.m[0][3] =
      (m.m[0][1] * m.m[1][3] * m.m[2][2] + m.m[0][2] * m.m[1][1] * m.m[2][3] +
       m.m[0][3] * m.m[1][2] * m.m[2][1] - m.m[0][1] * m.m[1][2] * m.m[2][3] -
       m.m[0][2] * m.m[1][3] * m.m[2][1] - m.m[0][3] * m.m[1][1] * m.m[2][2]) /
      det;
  r.m[1][0] =
      (m.m[1][0] * m.m[2][3] * m.m[3][2] + m.m[1][2] * m.m[2][0] * m.m[3][3] +
       m.m[1][3] * m.m[2][2] * m.m[3][0] - m.m[1][0] * m.m[2][2] * m.m[3][3] -
       m.m[1][2] * m.m[2][3] * m.m[3][0] - m.m[1][3] * m.m[2][0] * m.m[3][2]) /
      det;
  r.m[1][1] =
      (m.m[0][0] * m.m[2][2] * m.m[3][3] + m.m[0][2] * m.m[2][3] * m.m[3][0] +
       m.m[0][3] * m.m[2][0] * m.m[3][2] - m.m[0][0] * m.m[2][3] * m.m[3][2] -
       m.m[0][2] * m.m[2][0] * m.m[3][3] - m.m[0][3] * m.m[2][2] * m.m[3][0]) /
      det;
  r.m[1][2] =
      (m.m[0][0] * m.m[1][3] * m.m[3][2] + m.m[0][2] * m.m[1][0] * m.m[3][3] +
       m.m[0][3] * m.m[1][2] * m.m[3][0] - m.m[0][0] * m.m[1][2] * m.m[3][3] -
       m.m[0][2] * m.m[1][3] * m.m[3][0] - m.m[0][3] * m.m[1][0] * m.m[3][2]) /
      det;
  r.m[1][3] =
      (m.m[0][0] * m.m[1][2] * m.m[2][3] + m.m[0][2] * m.m[1][3] * m.m[2][0] +
       m.m[0][3] * m.m[1][0] * m.m[2][2] - m.m[0][0] * m.m[1][3] * m.m[2][2] -
       m.m[0][2] * m.m[1][0] * m.m[2][3] - m.m[0][3] * m.m[1][2] * m.m[2][0]) /
      det;
  r.m[2][0] =
      (m.m[1][0] * m.m[2][1] * m.m[3][3] + m.m[1][1] * m.m[2][3] * m.m[3][0] +
       m.m[1][3] * m.m[2][0] * m.m[3][1] - m.m[1][0] * m.m[2][3] * m.m[3][1] -
       m.m[1][1] * m.m[2][0] * m.m[3][3] - m.m[1][3] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[2][1] =
      (m.m[0][0] * m.m[2][3] * m.m[3][1] + m.m[0][1] * m.m[2][0] * m.m[3][3] +
       m.m[0][3] * m.m[2][1] * m.m[3][0] - m.m[0][0] * m.m[2][1] * m.m[3][3] -
       m.m[0][1] * m.m[2][3] * m.m[3][0] - m.m[0][3] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[2][2] =
      (m.m[0][0] * m.m[1][1] * m.m[3][3] + m.m[0][1] * m.m[1][3] * m.m[3][0] +
       m.m[0][3] * m.m[1][0] * m.m[3][1] - m.m[0][0] * m.m[1][3] * m.m[3][1] -
       m.m[0][1] * m.m[1][0] * m.m[3][3] - m.m[0][3] * m.m[1][1] * m.m[3][0]) /
      det;
  r.m[2][3] =
      (m.m[0][0] * m.m[1][3] * m.m[2][1] + m.m[0][1] * m.m[1][0] * m.m[2][3] +
       m.m[0][3] * m.m[1][1] * m.m[2][0] - m.m[0][0] * m.m[1][1] * m.m[2][3] -
       m.m[0][1] * m.m[1][3] * m.m[2][0] - m.m[0][3] * m.m[1][0] * m.m[2][1]) /
      det;
  r.m[3][0] =
      (m.m[1][0] * m.m[2][2] * m.m[3][1] + m.m[1][1] * m.m[2][0] * m.m[3][2] +
       m.m[1][2] * m.m[2][1] * m.m[3][0] - m.m[1][0] * m.m[2][1] * m.m[3][2] -
       m.m[1][1] * m.m[2][2] * m.m[3][0] - m.m[1][2] * m.m[2][0] * m.m[3][1]) /
      det;
  r.m[3][1] =
      (m.m[0][0] * m.m[2][1] * m.m[3][2] + m.m[0][1] * m.m[2][2] * m.m[3][0] +
       m.m[0][2] * m.m[2][0] * m.m[3][1] - m.m[0][0] * m.m[2][2] * m.m[3][1] -
       m.m[0][1] * m.m[2][0] * m.m[3][2] - m.m[0][2] * m.m[2][1] * m.m[3][0]) /
      det;
  r.m[3][2] =
      (m.m[0][0] * m.m[1][2] * m.m[3][1] + m.m[0][1] * m.m[1][0] * m.m[3][2] +
       m.m[0][2] * m.m[1][1] * m.m[3][0] - m.m[0][0] * m.m[1][1] * m.m[3][2] -
       m.m[0][1] * m.m[1][2] * m.m[3][0] - m.m[0][2] * m.m[1][0] * m.m[3][1]) /
      det;
  r.m[3][3] =
      (m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[0][1] * m.m[1][2] * m.m[2][0] +
       m.m[0][2] * m.m[1][0] * m.m[2][1] - m.m[0][0] * m.m[1][2] * m.m[2][1] -
       m.m[0][1] * m.m[1][0] * m.m[2][2] - m.m[0][2] * m.m[1][1] * m.m[2][0]) /
      det;

  return r;
}
template <typename T>
void decompose(const Matrix4x4<T> &m, Matrix4x4<T> &r, Matrix4x4<T> &s) {
  // extract rotation r from transformation matrix
  T norm;
  int count = 0;
  r = m;
  do {
    // compute next matrix in series
    Matrix4x4<T> Rnext;
    Matrix4x4<T> Rit = inverse(transpose(r));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        Rnext.m[i][j] = .5f * (r.m[i][j] + Rit.m[i][j]);
    // compute norm difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; i++) {
      T n = fabsf(r.m[i][0] - Rnext.m[i][0]) +
            fabsf(r.m[i][1] - Rnext.m[i][1]) + fabsf(r.m[i][2] - Rnext.m[i][2]);
      norm = std::max(norm, n);
    }
  } while (++count < 100 && norm > .0001f);
  // compute scale S using rotation and original matrix
  s = Matrix4x4<T>::mul(inverse(r), m);
}

template <typename T> class Matrix3x3 {
public:
  Matrix3x3() { memset(m, 0, sizeof(m)); }
  Matrix3x3(vec3 a, vec3 b, vec3 c)
      : Matrix3x3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z) {}
  Matrix3x3(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
  }
  void setIdentity() {
    memset(m, 0, sizeof(m));
    for (int i = 0; i < 3; i++)
      m[i][i] = 1.f;
  }
  T determinant() {
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
           m[0][2] * m[1][0] * m[2][1] - m[2][0] * m[1][1] * m[0][2] -
           m[2][1] * m[1][2] * m[0][0] - m[2][2] * m[1][0] * m[0][1];
  }
  T m[3][3];
};

template <typename T> Matrix3x3<T> operator*(T f, const Matrix3x3<T> &m) {
  return m * f;
}

typedef Matrix3x3<real_t> mat3;

template <typename T>
Vector3<T> operator*(const Matrix3x3<T> &A, const Vector3<T> &v) {
  Vector3<T> r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      r[i] += A.m[i][j] * v[j];
  return r;
}
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T> &A, const Matrix3x3<T> &B) {
  Matrix3x3<T> r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] =
          A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
  return r;
}
template <typename T>
Matrix3x3<T> operator*(const Matrix3x3<T> &A, const T &f) {
  return Matrix3x3<T>(A.m[0][0] * f, A.m[0][1] * f, A.m[0][2] * f, A.m[1][0] * f,
                   A.m[1][1] * f, A.m[1][2] * f, A.m[2][0] * f, A.m[2][1] * f,
                   A.m[2][2] * f);
}

template <typename T> Matrix3x3<T> inverse(const Matrix3x3<T> &m) {
  Matrix3x3<T> r;
  // r.setIdentity();
  T det =
      m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[1][0] * m.m[2][1] * m.m[0][2] +
      m.m[2][0] * m.m[0][1] * m.m[1][2] - m.m[0][0] * m.m[2][1] * m.m[1][2] -
      m.m[2][0] * m.m[1][1] * m.m[0][2] - m.m[1][0] * m.m[0][1] * m.m[2][2];
  if (std::fabs(det) < 1e-8)
    return r;
  r.m[0][0] = (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) / det;
  r.m[0][1] = (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) / det;
  r.m[0][2] = (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) / det;
  r.m[1][0] = (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) / det;
  r.m[1][1] = (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) / det;
  r.m[1][2] = (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]) / det;
  r.m[2][0] = (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) / det;
  r.m[2][1] = (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) / det;
  r.m[2][2] = (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) / det;
  return r;
}
template <typename T> Matrix3x3<T> transpose(const Matrix3x3<T> &m) {
  return Matrix3x3<T>(m.m[0][0], m.m[1][0], m.m[2][0], m.m[0][1], m.m[1][1],
                      m.m[2][1], m.m[0][2], m.m[1][2], m.m[2][2]);
}
template <typename T> Matrix3x3<T> star(const Vector3<T> a) {
  return Matrix3x3<T>(0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0);
}

template <typename T> class Matrix2x2 {
public:
  Matrix2x2() {
    memset(m, 0, sizeof(m));
    for (int i = 0; i < 2; i++)
      m[i][i] = 1.f;
  }
  Matrix2x2(T m00, T m01, T m10, T m11) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[1][0] = m10;
    m[1][1] = m11;
  }
  void setIdentity() {
    memset(m, 0, sizeof(m));
    for (int i = 0; i < 2; i++)
      m[i][i] = 1.f;
  }
  T determinant() { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }
  const T &operator()(u32 i, u32 j) const { return m[i][j]; }
  T &operator()(u32 i, u32 j) { return m[i][j]; }
  T m[2][2];
};

typedef Matrix2x2<real_t> mat2;

template <typename T>
Vector2<T> operator*(const Matrix2x2<T> &A, const Vector2<T> &v) {
  Vector2<T> r;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      r[i] += A.m[i][j] * v[j];
  return r;
}
template <typename T>
Matrix2x2<T> operator*(const Matrix2x2<T> &A, const Matrix2x2<T> &B) {
  Matrix2x2<T> r;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      r.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j];
  return r;
}
template <typename T> Matrix2x2<T> operator*(const Matrix2x2<T> &A, T f) {
  return Matrix2x2<T>(A.m[0][0] * f, A.m[0][1] * f, A.m[1][0] * f,
                      A.m[1][1] * f);
}
template <typename T> Matrix2x2<T> operator*(T f, const Matrix2x2<T> &m) {
  return m * f;
}

template <typename T> Matrix2x2<T> inverse(const Matrix2x2<T> &m) {
  Matrix2x2<T> r;
  T det = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];
  if (det == 0.f)
    return r;
  T k = 1.f / det;
  r.m[0][0] = m.m[1][1] * k;
  r.m[0][1] = -m.m[0][1] * k;
  r.m[1][0] = -m.m[1][0] * k;
  r.m[1][1] = m.m[0][0] * k;
  return r;
}

template <typename T> Matrix2x2<T> transpose(const Matrix2x2<T> &m) {
  return Matrix2x2<T>(m.m[0][0], m.m[1][0], m.m[0][1], m.m[1][1]);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix4x4<T> &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix3x3<T> &m) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix2x2<T> &m) {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++)
      os << m.m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

} // namespace ponos

#endif
