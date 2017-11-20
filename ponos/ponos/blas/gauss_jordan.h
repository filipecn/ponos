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

#ifndef PONOS_BLAS_GAUSS_JORDAN_H
#define PONOS_BLAS_GAUSS_JORDAN_H

#include <ponos/blas/blas.h>
#include <ponos/blas/linear_system.h>
#include <ponos/blas/solver_interface.h>

namespace ponos {

template <typename LinearSystemType>
class GaussJordanSolver : public LinearSystemSolver<LinearSystemType> {
public:
  GaussJordanSolver() {}
  bool solve(LinearSystemType *s) override {
    auto &A = s->A;
    auto &b = s->b;
    size_t n = A.rowsNumber(), irow = 0, icol = 0;
    std::vector<size_t> ipiv(n, 0), indxr(n), indxc(n);
    for (size_t i = 0; i < n; i++) {
      double big = 0;
      for (size_t j = 0; j < n; j++)
        if (ipiv[j] != 1)
          for (size_t k = 0; k < n; k++)
            if (ipiv[k] == 0) {
              if (std::fabs(A(j, k)) >= big) {
                big = std::fabs(A(j, k));
                irow = j;
                icol = k;
              }
            }
      ++(ipiv[icol]);
      if (irow != icol) {
        for (size_t l = 0; l < n; l++)
          std::swap(A(irow, l), A(icol, l));
        std::swap(b[irow], b[icol]);
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if (A(icol, icol) == 0.0)
        throw("gaussj: Singular Matrix");
      double pivinv = 1.0 / A(icol, icol);
      A(icol, icol) = 1.0;
      for (size_t l = 0; l < n; l++)
        A(icol, l) *= pivinv;
      b[icol] *= pivinv;
      for (size_t ll = 0; ll < n; ll++)
        if (ll != icol) {
          double dum = A(ll, icol);
          A(ll, icol) = 0.0;
          for (size_t l = 0; l < n; l++)
            A(ll, l) -= A(icol, l) * dum;
          b[ll] -= b[icol] * dum;
        }
    }
    for (int l = static_cast<int>(n) - 1; l >= 0; l--)
      if (indxr[l] != indxc[l])
        for (size_t k = 0; k < n; k++) {
          std::swap(A(k, indxr[l]), A(k, indxc[l]));
        }
    return true;
  }
};

} // ponos namespace

#endif // PONOS_BLAS_GAUSS_JORDAN_H
