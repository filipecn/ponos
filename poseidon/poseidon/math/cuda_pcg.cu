/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
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

#include <hermes/numeric/cuda_blas.h>
#include <poseidon/math/cuda_pcg.h>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

void pcg(MemoryBlock1Df &x, FDMatrix3D &A, MemoryBlock1Df &b,
         size_t maxNumberOfIterations, float *lastResidual) {
  MemoryBlock1Df m;
  MemoryBlock1Df r(b.size());
  r.allocate();
  MemoryBlock1Df d(b.size());
  d.allocate();
  MemoryBlock1Df q(b.size());
  q.allocate();
  fill1(q.accessor(), 0.f);
  MemoryBlock1Df s(b.size());
  s.allocate();
  fill1(s.accessor(), 0.f);
  // r = b - A * x
  mul(A, x, r);
  sub(b, r, r);
  // d = r
  memcpy(d, r);
  // sigma = r '* r
  double sigma = dot(r, r, m);
  double tolerance = 1e-6;
  uint it = 0;
  while (it < maxNumberOfIterations) {
    // q = Ad
    mul(A, d, q);
    // alpha = sigma / (d '* Ad)
    double alpha = sigma / dot(d, q, m);
    // x = alpha * d + x
    axpy(alpha, d, x, x);
    // r = r - alpha * Ad
    axpy(-alpha, q, r, r);
    // sigmaNew = r '* r
    double sigmaNew = dot(r, r, m);
    if (sigmaNew < tolerance * tolerance)
      break;
    // d = r + (sigmaNew / sigma) * d
    axpy(sigmaNew / sigma, d, r, d);
    sigma = sigmaNew;
    ++it;
  }
  *lastResidual = sigma;
}

} // namespace cuda

} // namespace poseidon