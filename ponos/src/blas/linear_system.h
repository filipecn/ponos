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

#ifndef PONOS_BLAS_LINEAR_SYSTEM_H
#define PONOS_BLAS_LINEAR_SYSTEM_H

#include "blas/fdm_matrix.h"
#include "blas/sparse_matrix.h"

namespace ponos {

template <typename MatrixType, typename VectorType> struct LinearSystem {
  LinearSystem() {}
  void resize(size_t w, size_t h) {}
  MatrixType A;
  VectorType x, b;
};

template <typename T> class LinearSystem<DenseMatrix<T>, DenseVector<T>> {
public:
  LinearSystem() {}
  void resize(size_t n, size_t m) {
    A.set(n, m);
    x.set(n);
    b.set(n);
  }
  DenseMatrix<T> A;
  DenseVector<T> x, b;
};

typedef LinearSystem<DenseMatrix<float>, DenseVector<float>> DenseLinearSystemf;

template <> class LinearSystem<FDMMatrix2Df, FDMVector2Df> {
public:
  LinearSystem() {}
  void resize(size_t w, size_t h) {
    A.resize(w, h);
    x.resize(w, h);
    b.resize(w, h);
  }
  FDMMatrix2Df A;
  FDMVector2Df x, b;
};

typedef LinearSystem<FDMMatrix2Df, FDMVector2Df> FDMLinearSystem2f;

template <> class LinearSystem<SparseMatrix<double>, SparseVector<double>> {
public:
  LinearSystem() {}
  void resize(size_t w, size_t h) {}
  SparseMatrix<double> A;
  SparseVector<double> x, b;
};

typedef LinearSystem<SparseMatrix<double>, SparseVector<double>>
    SparseLinearSystemd;

} // ponos namespace

#endif // PONOS_BLAS_LINEAR_SYSTEM_H
