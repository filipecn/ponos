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

#ifndef PONOS_BLAS_PCG_H
#define PONOS_BLAS_PCG_H

#include <cmath>

namespace ponos {

template <typename BlasType> struct NullCGPreconditioner {
  void build(const typename BlasType::MatrixType &) {}
  void solve(const typename BlasType::VectorType &b,
             typename BlasType::VectorType *x) {
    BlasType::set(b, x);
  }
};

template <typename BlasType> struct IncompleteCholeskyCGPreconditioner {
  typename BlasType::VectorType d;
  typename BlasType::VectorType y;
  const typename BlasType::MatrixType *A;
  IncompleteCholeskyCGPreconditioner() {}
  void build(const typename BlasType::MatrixType &m) {
    A = &m;
    d.resize(A->width, A->height, 0.0);
    y.resize(A->width, A->height, 0.0);
    for (size_t i = 0; i < A->width; i++)
      for (size_t j = 0; j < A->height; j++) {
        double denominator =
            (*A)(i, j).center -
            ((i > 0) ? SQR((*A)(i - 1, j).right) * d(i - 1, j) : 0.0) -
            ((j > 0) ? SQR((*A)(i, j - 1).up) * d(i, j - 1) : 0.0);
        if (std::fabs(denominator) > 0.0)
          d(i, j) = 1.f / denominator;
        else
          d(i, j) = 0.0;
      }
  }
  void solve(const typename BlasType::VectorType &b,
             typename BlasType::VectorType *x) {
    size_t w = b.width, h = b.height;
    for (size_t i = 0; i < w; i++)
      for (size_t j = 0; j < h; j++)
        y(i, j) =
            (b(i, j) - ((i > 0) ? (*A)(i - 1, j).right *y(i - 1, j) : 0.0) -
             ((j > 0) ? (*A)(i, j - 1).up *y(i, j - 1) : 0.0)) *
            d(i, j);
    for (size_t i = w - 1; i >= 0; i--)
      for (size_t j = h - 1; j >= 0; j--)
        (*x)(i, j) =
            (y(i, j) - ((i + 1 < w) ? (*A)(i, j).right *(*x)(i + 1, j) : 0.0) -
             ((j + 1 < h) ? (*A)(i, j).up *(*x)(i, j + 1) : 0.0)) *
            d(i, j);
  }
};

} // ponos namespace

#endif // PONOS_BLAS_PCG_H
