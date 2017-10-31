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

#ifndef PONOS_BLAS_SPARSE_GAUSS_SEIDEL_SOLVER_H
#define PONOS_BLAS_SPARSE_GAUSS_SEIDEL_SOLVER_H

#include "blas/blas.h"
#include "blas/linear_system.h"
#include "blas/solver_interface.h"

namespace ponos {

class SparseGaussSeidelSolverd
    : public LinearSystemSolver<SparseLinearSystemd> {
public:
  SparseGaussSeidelSolverd(uint _maxNumberOfIterations, double _tolerance,
                           uint _residualCheckInterval)
      : maxNumberOfIterations(_maxNumberOfIterations), tolerance(_tolerance),
        residualCheckInterval(_residualCheckInterval) {}

  bool solve(SparseLinearSystemd *s) override {
    for (uint it = 0; it < maxNumberOfIterations; it++) {
      lastNumberOfIterations = it;
      relax(s, &xTmp);
      xTmp.swap(&s->x);
      if (it != 0 && it % residualCheckInterval == 0) {
        SparseBlas2d::residual(s->A, s->x, s->b, &residual);
        // if (SparseBlas2d::l2Norm(residual) < tolerance) {
        if (SparseBlas2d::infNorm(s->x, xTmp) < tolerance) {
          break;
        }
      }
    }
    SparseBlas2d::residual(s->A, s->x, s->b, &residual);
    residual.iterate([](const double &v, size_t i) {
      UNUSED_VARIABLE(i);
      std::cout << "---> " << v << std::endl;
    });
    // std::cout << residual.elementCount() << std::endl;
    // lastResidual = SparseBlas2d::l2Norm(residual);
    lastResidual = SparseBlas2d::infNorm(s->x, xTmp);
    return lastResidual < tolerance;
  }

  uint lastNumberOfIterations;
  double lastResidual;

private:
  void relax(SparseLinearSystemd *s, SparseVector<double> *xt) {
    SparseMatrix<double> &A = s->A;
    SparseVector<double> &x = s->x;
    SparseVector<double> &b = s->b;
    xt->resetCount();
    SparseVector<double>::const_iterator bit(b);
    for (size_t i = 0; i < A.rowCount(); i++) {
      bool nonzero = false;
      double aij = 1.0, sum = 0.0;
      SparseVector<double>::const_iterator xit(x);
      SparseVector<double>::const_iterator xtit(*xt);
      A.iterateRow(i, [&](const double &v, size_t j) {
        if (i == j) {
          aij = v;
          return;
        }
        if (i > j) {
          if (xtit.jumpTo(j)) {
            sum += v * xtit.value();
            nonzero = true;
          }
        } else {
          if (xit.jumpTo(j)) {
            sum += v * xit.value();
            nonzero = true;
          }
        }
      });
      if (bit.jumpTo(i)) {
        sum = bit.value() - sum;
        nonzero = true;
      } else
        sum = -sum;
      if (nonzero)
        xt->insert(i, sum / aij);
    }
  }

  uint maxNumberOfIterations;
  double tolerance;
  uint residualCheckInterval;

  SparseVector<double> xTmp;
  SparseVector<double> residual;
};

} // ponos namespace

#endif // PONOS_BLAS_SPARSE_GAUSS_SEIDEL_SOLVER_H
