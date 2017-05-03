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

#include "blas/fdm_matrix.h"
#include <cmath>

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
  static void set(float s, FDMMatrix2Df *result) {}
  // Sets the given matrix to the output matrix.
  static void set(const FDMMatrix2Df &m, FDMMatrix2Df *result) {}
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

} // ponos namespace

#endif // PONOS_BLAS_BLAS_H
