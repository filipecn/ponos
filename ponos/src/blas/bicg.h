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

#ifndef PONOS_BLAS_BICG_H
#define PONOS_BLAS_BICG_H

#include "blas/custom_matrix.h"

namespace ponos {

template <typename M, typename T>
void bicg(const MatrixConstAccessor<M, T> &A,
          const MatrixConstAccessor<M, T> &At, VectorInterface<T> *x,
          const VectorInterface<T> *b, VectorInterface<T> *r,
          VectorInterface<T> *rt, VectorInterface<T> *p, VectorInterface<T> *pt,
          VectorInterface<T> *z, VectorInterface<T> *zt,
          size_t maxNumberOfIterations, double tolerance, int itol) {
  size_t n = x->size();
  T bnrm, znrm, xnrm, zm1nrm, dxnrm, bkden = 1.0, bk, err;
  residual(A, x, b, r);
  for (size_t j = 0; j < n; j++)
    (*rt)[j] = (*r)[j];
  /* {
     std::cout << "r0\n";
     for (size_t i = 0; i < n; i++)
       std::cout << (*r)[i] << " ";
     std::cout << std::endl;
   }*/
  if (itol == 1) {
    bnrm = norm(b);
    dSolve(A, r, z);
  } else if (itol == 2) {
    dSolve(A, b, z);
    bnrm = norm(z);
    dSolve(A, r, z);
  } else {
    dSolve(A, b, z);
    bnrm = (itol == 3) ? norm(z) : infnorm(z);
    dSolve(A, r, z);
    znrm = (itol == 3) ? norm(z) : infnorm(z);
  }
  /* {
     std::cout << "z0\n";
     for (size_t i = 0; i < n; i++)
       std::cout << (*z)[i] << " ";
     std::cout << std::endl;
   }*/
  for (size_t iter = 0; iter < maxNumberOfIterations; ++iter) {
    dSolve(At, rt, zt);
    double bknum = dot(rt, z);
    if (iter == 0) {
      for (size_t j = 0; j < n; j++) {
        (*p)[j] = (*z)[j];
        (*pt)[j] = (*zt)[j];
      }
    } else {
      bk = bknum / bkden;
      axpy(bk, p, z, p);
      axpy(bk, pt, zt, pt);
    }
    bkden = bknum;
    mvm(A, p, z);
    double akden = dot(z, pt);
    double ak = bknum / akden;
    mvm(At, pt, zt);
    axpy(static_cast<T>(ak), p, x, x);
    axpy(static_cast<T>(-ak), z, r, r);
    axpy(static_cast<T>(-ak), zt, rt, rt);
    dSolve(A, r, z);
    if (itol == 1)
      err = norm(r) / bnrm;
    else if (itol == 2)
      err = norm(z) / bnrm;
    else {
      zm1nrm = znrm;
      znrm = (itol == 3) ? norm(z) : infnorm(z);
      if (fabs(zm1nrm - znrm) > tolerance * znrm) {
        dxnrm = fabs(ak) * ((itol == 3) ? norm(p) : infnorm(p));
        err = znrm / fabs(zm1nrm - znrm) * dxnrm;
      } else {
        err = znrm / bnrm;
        continue;
      }
      xnrm = (itol == 3) ? norm(x) : infnorm(x);
      if (err <= 0.5 * xnrm)
        err /= xnrm;
      else {
        err = znrm / bnrm;
        continue;
      }
    }
    if (err <= tolerance)
      break;
  }
}

} // ponos namespace

#endif // PONOS_BLAS_BICG_H
