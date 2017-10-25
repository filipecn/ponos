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

#ifndef PONOS_BLAS_CG_H
#define PONOS_BLAS_CG_H

namespace ponos {

template <typename BlasType, typename PrecondType>
void pcg(const typename BlasType::MatrixType &A,
         const typename BlasType::VectorType &b,
         unsigned int maxNumberOfIterations, double tolerance, PrecondType *M,
         typename BlasType::VectorType *x, typename BlasType::VectorType *r,
         typename BlasType::VectorType *d, typename BlasType::VectorType *q,
         typename BlasType::VectorType *s, unsigned int *lastNumberOfIterations,
         double *lastResidual) {
  UNUSED_VARIABLE(M);
  BlasType::set(0, r);
  BlasType::set(0, d);
  BlasType::set(0, q);
  BlasType::set(0, s);
  // r = b - A * x
  BlasType::residual(A, *x, b, r);
  // d = r
  BlasType::set(*r, d);
  // sigma = r '* r
  double sigma = BlasType::dot(*r, *r);
  uint it = 0;
  while (it < maxNumberOfIterations) {
    // q = Ad
    BlasType::mvm(A, *d, q);
    // alpha = sigma / (d '* Ad)
    double alpha = sigma / BlasType::dot(*d, *q);
    // x = alpha * d + x
    BlasType::axpy(alpha, *d, *x, x);
    // r = r - alpha * Ad
    BlasType::axpy(-alpha, *q, *r, r);
    // sigmaNew = r '* r
    double sigmaNew = BlasType::dot(*r, *r);
    if (sigmaNew < SQR(tolerance))
      break;
    // d = r + (sigmaNew / sigma) * d
    BlasType::axpy(sigmaNew / sigma, *d, *r, d);
    sigma = sigmaNew;
    ++it;
  }

  *lastNumberOfIterations = it;
  *lastResidual = sigma;
}

template <typename BlasType, typename PrecondType>
void cg(const typename BlasType::MatrixType &A,
        const typename BlasType::VectorType &b,
        unsigned int maxNumberOfIterations, double tolerance,
        typename BlasType::VectorType *x, typename BlasType::VectorType *r,
        typename BlasType::VectorType *d, typename BlasType::VectorType *q,
        typename BlasType::VectorType *s, unsigned int *lastNumberOfIterations,
        double *lastResidual) {
  PrecondType p;
  pcg<BlasType, PrecondType>(A, b, maxNumberOfIterations, tolerance, &p, x, r,
                             d, q, s, lastNumberOfIterations, lastResidual);
}

} // ponos namespace

#endif // PONOS_BLAS_CG_H
