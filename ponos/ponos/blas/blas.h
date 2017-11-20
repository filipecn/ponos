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

#ifndef PONOS_BLAS_BLAS_H
#define PONOS_BLAS_BLAS_H

#include <cmath>
#include <ponos/blas/fdm_matrix.h>
#include <ponos/blas/sparse_matrix.h>
#include <ponos/blas/sparse_vector.h>

namespace ponos {

template <typename S, class V, class M> struct Blas {
  typedef S ScalarType;
  typedef V VectorType;
  typedef M MatrixType;
  // Sets the given scalar value to the output vector.
  static void set(ScalarType s, VectorType *result);
  // Sets the given vector to the output vector.
  static void set(const VectorType &v, VectorType *result);
  // Sets the given scalar value to the output matrix.
  static void set(ScalarType s, MatrixType *result);
  // Sets the given matrix to the output matrix.
  static void set(const MatrixType &m, MatrixType *result);
  // Performs dot product.
  static ScalarType dot(const VectorType &a, const VectorType &b);
  // Calculates a*x + y.
  static void axpy(ScalarType a, const VectorType &x, const VectorType &y,
                   VectorType *result);
  // Performs matrix-vector multiplication.
  static void mvm(const MatrixType &m, const VectorType &v, VectorType *result);
  // Calculates b - A*x.
  static void residual(const MatrixType &a, const VectorType &x,
                       const VectorType &b, VectorType *result);
  // Returns the length of the vector.
  static ScalarType l2Norm(const VectorType &v);
};

typedef Blas<float, FDMVector2Df, FDMMatrix2Df> FDMBlas2f;

template <> struct Blas<float, FDMVector2Df, FDMMatrix2Df> {
  typedef float ScalarType;
  typedef FDMVector2Df VectorType;
  typedef FDMMatrix2Df MatrixType;
  // Sets the given scalar value to the output vector.
  static void set(float s, FDMVector2Df *result) {
    for (size_t i = 0; i < result->width; i++)
      for (size_t j = 0; j < result->height; j++)
        (*result)(i, j) = s;
  }
  // Sets the given vector to the output vector.
  static void set(const FDMVector2Df &v, FDMVector2Df *result) {
    for (size_t i = 0; i < result->width; i++)
      for (size_t j = 0; j < result->height; j++)
        (*result)(i, j) = v(i, j);
  }
  // Sets the given scalar value to the output matrix.
  static void set(float s, FDMMatrix2Df *result) {
    UNUSED_VARIABLE(s);
    UNUSED_VARIABLE(result);
  }
  // Sets the given matrix to the output matrix.
  static void set(const FDMMatrix2Df &m, FDMMatrix2Df *result) {
    UNUSED_VARIABLE(m);
    UNUSED_VARIABLE(result);
  }
  // Performs dot product.
  static float dot(const FDMVector2Df &a, const FDMVector2Df &b) {
    float sum = 0.f;
    for (size_t i = 0; i < a.width; i++)
      for (size_t j = 0; j < a.height; j++)
        sum += a(i, j) * b(i, j);
    return sum;
  }
  // Calculates a*x + y.
  static void axpy(float a, const FDMVector2Df &x, const FDMVector2Df &y,
                   FDMVector2Df *result) {
    for (size_t i = 0; i < result->width; i++)
      for (size_t j = 0; j < result->height; j++)
        (*result)(i, j) = x(i, j) * a + y(i, j);
  }
  // Performs matrix-vector multiplication.
  static void mvm(const FDMMatrix2Df &m, const FDMVector2Df &v,
                  FDMVector2Df *result) {
    size_t w = result->width;
    size_t h = result->height;
    for (size_t i = 0; i < w; i++)
      for (size_t j = 0; j < h; j++)
        (*result)(i, j) = m(i, j).center * v(i, j) +
                          ((i > 0) ? m(i - 1, j).right * v(i - 1, j) : 0.f) +
                          ((i + 1 < w) ? m(i, j).right * v(i + 1, j) : 0.f) +
                          ((j > 0) ? m(i, j - 1).up * v(i, j - 1) : 0.f) +
                          ((j + 1 < h) ? m(i, j).up * v(i, j + 1) : 0.f);
  }
  // Calculates b - A*x.
  static void residual(const FDMMatrix2Df &a, const FDMVector2Df &x,
                       const FDMVector2Df &b, FDMVector2Df *result) {
    mvm(a, x, result);
    axpy(-1.f, *result, b, result);
  }
  // Returns the length of the vector.
  static float l2Norm(const FDMVector2Df &v) {
    float sum = 0.f;
    for (size_t i = 0; i < v.width; i++)
      for (size_t j = 0; j < v.height; j++)
        sum += v(i, j) * v(i, j);
    return sqrtf(sum);
  }
};

typedef Blas<double, SparseVector<double>, SparseMatrix<double>> SparseBlas2d;

template <> struct Blas<double, SparseVector<double>, SparseMatrix<double>> {
  typedef double ScalarType;
  typedef SparseVector<double> VectorType;
  typedef SparseMatrix<double> MatrixType;
  // Sets the given scalar value to the output vector.
  static void set(ScalarType s, VectorType *result) {
    if (IS_EQUAL_ERROR(s, static_cast<ScalarType>(0.0),
                       static_cast<ScalarType>(1e-8)))
      result->resetCount();
    else
      result->iterate([s](ScalarType &v, size_t i) {
        UNUSED_VARIABLE(i);
        v = s;
      });
  }
  // Sets the given vector to the output vector.
  static void set(const VectorType &v, VectorType *result) { result->copy(v); }
  // Sets the given scalar value to the output matrix.
  static void set(ScalarType s, MatrixType *result) {
    for (size_t r = 0; r < result->rowCount(); r++)
      result->iterateRow(r, [s](ScalarType &v, size_t j) {
        UNUSED_VARIABLE(j);
        v = s;
      });
  }
  // Sets the given matrix to the output matrix.
  static void set(const MatrixType &m, MatrixType *result) { result->copy(m); }
  // Performs dot product.
  static ScalarType dot(const VectorType &a, const VectorType &b) {
    return a.dot(b);
  }
  // Calculates a*x + y.
  static void axpy(ScalarType a, const VectorType &x, const VectorType &y,
                   VectorType *result) {
    VectorType tmp;
    ::ponos::axpy(a, x, y, &tmp);
    result->copy(tmp);
  }
  // Performs matrix-vector multiplication.
  static void mvm(const MatrixType &m, const VectorType &v,
                  VectorType *result) {
    result->resetCount();
    for (size_t r = 0; r < m.rowCount(); r++) {
      VectorType::const_iterator it(v);
      ScalarType sum = 0.0;
      bool nonZero = false;
      m.iterateRow(r, [&](const ScalarType &v, size_t j) {
        if (it.rowIndex() < j)
          while (it.next() && it.rowIndex() < j)
            ++it;
        if (!it.next())
          return;
        if (it.rowIndex() == j) {
          nonZero = true;
          sum += v * it.value();
        }
      });
      if (nonZero)
        result->insert(r, sum);
    }
  }
  // Calculates b - A*x.
  static void residual(const MatrixType &a, const VectorType &x,
                       const VectorType &b, VectorType *result) {
    VectorType tmp;
    mvm(a, x, &tmp);
    ::ponos::axpy(-1.0, tmp, b, result);
  }
  // Returns the length of the vector.
  static ScalarType l2Norm(const VectorType &v) {
    ScalarType sum = 0.0;
    for (VectorType::const_iterator it(v); it.next(); ++it)
      sum += it.value() * it.value();
    return sqrt(sum);
  }
  static ScalarType infNorm(const VectorType &av, const VectorType &bv) {
    ScalarType m = 0.0;
    VectorType::const_iterator a(av);
    VectorType::const_iterator b(bv);
    while (a.next() || b.next()) {
      ScalarType d = 0.0;
      if ((a.next() && !b.next()) ||
          (a.next() && b.next() && a.rowIndex() < b.rowIndex()))
        d = std::fabs(a.value()), ++a;
      else if ((!a.next() && b.next()) ||
               (a.next() && b.next() && b.rowIndex() < a.rowIndex()))
        d = std::fabs(b.value()), ++b;
      else
        d = std::fabs(a.value() - b.value()), ++b, ++a;
      m = std::max(m, d);
    }
    return m;
  }

  static ScalarType infNorm(const VectorType &v) {
    ScalarType m = 0.0;
    VectorType::const_iterator a(v);
    while (a.next()) {
      m = std::max(m, a.value());
      ++a;
    }
    return m;
  }
};

} // ponos namespace

#endif // PONOS_BLAS_BLAS_H
