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
class Matrix4x4 {
public:
  /// \param isIdentity [optional | def = true] initialize as an identity matrix
  explicit Matrix4x4(bool isIdentity = true);
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  Matrix4x4(std::initializer_list<float> values, bool columnMajor = false);
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  explicit Matrix4x4(const float mat[16], bool columnMajor = false);
  /// \param mat matrix entries in [ROW][COLUMN] form
  explicit Matrix4x4(float mat[4][4]);
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
  Matrix4x4(float m00, float m01, float m02, float m03, float m10, float m11,
            float m12, float m13, float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33);
  void setIdentity();
  void row_major(float *a) const {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        a[k++] = m[i][j];
  }
  void column_major(float *a) const {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        a[k++] = m[j][i];
  }
  Matrix4x4 operator*(const Matrix4x4 &mat) {
    Matrix4x4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        r.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] +
                    m[i][2] * mat.m[2][j] + m[i][3] * mat.m[3][j];
    return r;
  }
  Vector4 operator*(const Vector4 &v) const {
    Vector4 r;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r[i] += m[i][j] * v[j];
    return r;
  }
  bool operator==(const Matrix4x4 &_m) const {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (!IS_EQUAL(m[i][j], _m.m[i][j]))
          return false;
    return true;
  }
  bool operator!=(const Matrix4x4 &_m) const { return !(*this == _m); }
  bool isIdentity() {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (i != j && !IS_EQUAL(m[i][j], 0.f))
          return false;
        else if (i == j && !IS_EQUAL(m[i][j], 1.f))
          return false;
    return true;
  }
  static Matrix4x4 mul(const Matrix4x4 &m1, const Matrix4x4 &m2) {
    Matrix4x4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                    m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
    return r;
  }
  friend std::ostream &operator<<(std::ostream &os, const Matrix4x4 &m);
  float m[4][4];
};

Matrix4x4 transpose(const Matrix4x4 &m);
Matrix4x4 inverse(const Matrix4x4 &m);
void decompose(const Matrix4x4 &m, Matrix4x4 &r, Matrix4x4 &s);
typedef Matrix4x4 mat4;

class Matrix3x3 {
public:
  Matrix3x3();
  Matrix3x3(vec3 a, vec3 b, vec3 c);
  Matrix3x3(float m00, float m01, float m02, float m10, float m11, float m12,
            float m20, float m21, float m22);
  void setIdentity();
  Vector3 operator*(const Vector3 &v) const {
    Vector3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r[i] += m[i][j] * v[j];
    return r;
  }
  Matrix3x3 operator*(const Matrix3x3 &mat) {
    Matrix3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] +
                    m[i][2] * mat.m[2][j];
    return r;
  }
  Matrix3x3 operator*(const float &f) const {
    return Matrix3x3(m[0][0] * f, m[0][1] * f, m[0][2] * f, m[1][0] * f,
                     m[1][1] * f, m[1][2] * f, m[2][0] * f, m[2][1] * f,
                     m[2][2] * f);
  }
  static Matrix3x3 mul(const Matrix3x3 &m1, const Matrix3x3 &m2) {
    Matrix3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
                    m1.m[i][2] * m2.m[2][j];
    return r;
  }
  float determinant() {
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
           m[0][2] * m[1][0] * m[2][1] - m[2][0] * m[1][1] * m[0][2] -
           m[2][1] * m[1][2] * m[0][0] - m[2][2] * m[1][0] * m[0][1];
  }
  friend std::ostream &operator<<(std::ostream &os, const Matrix3x3 &m);
  float m[3][3];
};

inline Matrix3x3 operator*(float f, const Matrix3x3 &m) { return m * f; }

Matrix3x3 transpose(const Matrix3x3 &m);
Matrix3x3 inverse(const Matrix3x3 &m);
Matrix3x3 star(const Vector3 a);

typedef Matrix3x3 mat3;

class Matrix2x2 {
public:
  Matrix2x2();
  Matrix2x2(float m00, float m01, float m10, float m11);
  void setIdentity();
  float determinant() { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }
  Vector2 operator*(const Vector2 &v) const {
    Vector2 r;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        r[i] += m[i][j] * v[j];
    return r;
  }
  Matrix2x2 operator*(const Matrix2x2 &mat) {
    Matrix2x2 r;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        r.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j];
    return r;
  }
  Matrix2x2 operator*(const float &f) const {
    return Matrix2x2(m[0][0] * f, m[0][1] * f, m[1][0] * f, m[1][1] * f);
  }
  static Matrix2x2 mul(const Matrix2x2 &m1, const Matrix2x2 &m2) {
    Matrix2x2 r;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j];
    return r;
  }
  friend std::ostream &operator<<(std::ostream &os, const Matrix2x2 &m);
  float m[2][2];
};

inline Matrix2x2 operator*(float f, const Matrix2x2 &m) { return m * f; }

Matrix2x2 transpose(const Matrix2x2 &m);
Matrix2x2 inverse(const Matrix2x2 &m);

typedef Matrix2x2 mat2;

} // ponos namespace

#endif
