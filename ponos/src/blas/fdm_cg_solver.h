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

#ifndef PONOS_BLAS_CG_SOLVER_H
#define PONOS_BLAS_CG_SOLVER_H

#include "blas/blas.h"
#include "blas/linear_system.h"
#include "blas/pcg.h"
#include "blas/solver_interface.h"

namespace ponos {

template <typename Preconditioner = IncompleteCholeskyCGPreconditioner>
class FDMCGSolver2f : public LinearSystemSolver<FDMLinearSystem2f> {
public:
  bool solve(FDMLinearSystem2f *s) override {
    FDMMatrix2Df &matrix = s->A;
    FDMVector2Df &solution = s->x;
    FDMVector2Df &rhs = s->b;

    r.resize(matrix.width, matrix.height);
    d.resize(matrix.width, matrix.height);
    q.resize(matrix.width, matrix.height);
    s.resize(matrix.width, matrix.height);

    s->x.set(0.0);
    r.set(0.0);
    d.set(0.0);
    q.set(0.0);
    s.set(0.0);

    precond.build(matrix);

    pcg<FDMBlas2f, Preconditioner>(matrix, rhs, maxNumberOfIterations,
                                   tolerance, &precond, &solution, &r, &d, &q,
                                   &s, &lastNumberOfIterations, &lastResidual);

    return lastResidual <= tolerance ||
           lastNumberOfIterations < maxNumberOfIterations;
  }

private:
  uint maxNumberOfIterations;
  uint lastNumberOfIterations;
  uint residualCheckInterval;
  double tolerance;
  double lastResidual;

  FDMVector2Df r, d, q, s;
  Preconditioner<FDMBlas2f> precond;
};

} // ponos namespace

#endif // PONOS_BLAS_CG_SOLVER_H
