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

void pcg(MemoryBlock1Dd &x, FDMatrix2D &A, MemoryBlock1Dd &b,
         size_t maxNumberOfIterations, float tolerance) {
  // cpu memory
  MemoryBlock1Hd h_r(b.size(), 0);
  MemoryBlock1Hd h_z(b.size(), 0);
  MemoryBlock1Hd precon(b.size(), 0);
  // FDMatrix2H h_A(A.gridSize());
  // h_A.copy(A);
  // mic0(precon, h_A, 0.97, 0.25);
  // device memory
  MemoryBlock1Dd m;
  MemoryBlock1Dd r(b.size(), 0); // residual
  MemoryBlock1Dd z(b.size(), 0); // auxiliar
  MemoryBlock1Dd s(b.size(), 0); // search
  std::cerr << "max " << maxNumberOfIterations << std::endl;
  // r = b - A * x
  mul(A, x, r);
  sub(b, r, r);
  if (infnorm(r, m) <= tolerance)
    return;
  // z = M * r
  memcpy(z, r);
  // memcpy(h_r, r);
  // memcpy(h_z, z);
  // applyMIC0(h_A, precon, h_r, h_z);
  // memcpy(z, h_z);
  // s = z
  memcpy(s, z);
  // sigma = z '* r
  double sigma = dot(z, r, m);
  // std::cerr << "sigma = " << sigma << std::endl;
  size_t it = 0;
  // std::cerr << "S " << s << std::endl;
  while (it < maxNumberOfIterations) {
    // z = As
    mul(A, s, z);
    // std::cerr << "Z " << z << std::endl;
    // std::cerr << "S " << s << std::endl;
    // alpha = sigma / (z '* s)
    double alpha = sigma / dot(z, s, m);
    // std::cerr << "alpha " << alpha << std::endl;
    // x = alpha * s + x
    axpy(alpha, s, x, x);
    // r = r - alpha * z
    axpy(-alpha, z, r, r);
    // std::cerr << "r norm test\n";
    // std::cerr << r << std::endl;
    if (infnorm(r, m) <= tolerance) {
      std::cerr << "PCG RUN with " << it << " iterations.\n";
      return;
    }
    // z = M * r
    memcpy(z, r);
    // memcpy(h_r, r);
    // memcpy(h_z, z);
    // applyMIC0(h_A, precon, h_r, h_z);
    // memcpy(z, h_z);
    // sigmaNew = z '* r
    double sigmaNew = dot(z, r, m);
    // std::cerr << "sigmaNew " << sigmaNew << std::endl;
    // if (sigmaNew < tolerance * tolerance)
    //   break;
    // s = z + (sigmaNew / sigma) * s
    axpy(sigmaNew / sigma, s, z, s);
    sigma = sigmaNew;
    ++it;
  }
  // auto acc = h_A.accessor();
  // std::cerr << "BAD PCG!\n" << acc << std::endl;
  // std::cerr << b << std::endl;
  exit(-1);
}

void mic0(MemoryBlock1Hd &precon, FDMatrix3H &h_A, double tal, double sigma) {
#define SQR(A) ((A) * (A))
  auto A = h_A.accessor();
  auto E = precon.accessor();
  auto size = h_A.gridSize();
  auto indices = h_A.indexDataAccessor();
  for (size_t k = 0; k < size.z; k++)
    for (size_t j = 0; j < size.y; j++)
      for (size_t i = 0; i < size.x; i++)
        if (indices(i, j, k) >= 0) {
          double e =
              A(i, j, k, i, j, k) -
              ((indices.isIndexValid(i - 1, j, k) && indices(i - 1, j, k) >= 0)
                   ? SQR(A(i - 1, j, k, i, j, k) * E[indices(i - 1, j, k)])
                   : 0.0) -
              ((indices.isIndexValid(i, j - 1, k) && indices(i, j - 1, k) >= 0)
                   ? SQR(A(i, j - 1, k, i, j, k) * E[indices(i, j - 1, k)])
                   : 0.0) -
              ((indices.isIndexValid(i, j, k - 1) && indices(i, j, k - 1) >= 0)
                   ? SQR(A(i, j, k - 1, i, j, k) * E[indices(i, j, k - 1)])
                   : 0.0) -
              tal * (A(i - 1, j, k, i, j, k) *
                         (A(i - 1, j, k, i - 1, j + 1, k) +
                          A(i - 1, j, k, i - 1, j, k + 1)) *
                         ((indices.isIndexValid(i - 1, j, k) &&
                           indices(i - 1, j, k) >= 0)
                              ? SQR(E[indices(i - 1, j, k)])
                              : 0.0) +
                     A(i, j - 1, k, i, j, k) *
                         (A(i, j - 1, k, i + 1, j - 1, k) +
                          A(i, j - 1, k, i, j - 1, k + 1)) *
                         ((indices.isIndexValid(i, j - 1, k) &&
                           indices(i, j - 1, k) >= 0)
                              ? SQR(E[indices(i, j - 1, k)])
                              : 0.0) +
                     A(i, j, k - 1, i, j, k) *
                         (A(i, j, k - 1, i + 1, j, k - 1) +
                          A(i, j, k - 1, i, j + 1, k - 1)) *
                         ((indices.isIndexValid(i, j, k - 1) &&
                           indices(i, j, k - 1) >= 0)
                              ? SQR(E[indices(i, j, k - 1)])
                              : 0.0));
          if (e < sigma * A(i, j, k, i, j, k))
            e = A(i, j, k, i, j, k);
          E[indices(i, j, k)] = 1 / sqrt(e);
        }
}

void applyMIC0(FDMatrix3H &h_A, MemoryBlock1Hd &h_precon, MemoryBlock1Hd &h_r,
               MemoryBlock1Hd &h_z) {
  MemoryBlock1Hd h_q(h_z.size(), 0);
  auto indices = h_A.indexDataAccessor();
  auto A = h_A.accessor();
  auto precon = h_precon.accessor();
  auto r = h_r.accessor();
  auto z = h_z.accessor();
  auto q = h_q.accessor();
  auto size = h_A.gridSize();
  // solve Lq = r
  for (size_t k = 0; k < size.z; ++k)
    for (size_t j = 0; j < size.y; ++j)
      for (size_t i = 0; i < size.x; ++i)
        if (indices(i, j, k) >= 0) {
          double t =
              r[indices(i, j, k)] -
              ((indices.isIndexValid(i - 1, j, k) && indices(i - 1, j, k) >= 0)
                   ? A(i - 1, j, k, i, j, k) * precon[indices(i - 1, j, k)] *
                         q[indices(i - 1, j, k)]
                   : 0.0) -
              ((indices.isIndexValid(i, j - 1, k) && indices(i, j - 1, k) >= 0)
                   ? A(i, j - 1, k, i, j, k) * precon[indices(i, j - 1, k)] *
                         q[indices(i, j - 1, k)]
                   : 0.0) -
              ((indices.isIndexValid(i, j, k - 1) && indices(i, j, k - 1) >= 0)
                   ? A(i, j, k - 1, i, j, k) * precon[indices(i, j, k - 1)] *
                         q[indices(i, j, k - 1)]
                   : 0.0);
          q[indices(i, j, k)] = t * precon[indices(i, j, k)];
        }
  // solve L^Tz = q
  for (int k = (int)size.z - 1; k >= 0; --k)
    for (int j = (int)size.y - 1; j >= 0; --j)
      for (int i = (int)size.x - 1; i >= 0; --i)
        if (indices(i, j, k) >= 0) {
          double t =
              q[indices(i, j, k)] -
              ((indices.isIndexValid(i + 1, j, k) && indices(i + 1, j, k) >= 0)
                   ? A(i, j, k, i + 1, j, k) * precon[indices(i, j, k)] *
                         z[indices(i + 1, j, k)]
                   : 0.0) -
              ((indices.isIndexValid(i, j + 1, k) && indices(i, j + 1, k) >= 0)
                   ? A(i, j, k, i, j + 1, k) * precon[indices(i, j, k)] *
                         z[indices(i, j + 1, k)]
                   : 0.0) -
              ((indices.isIndexValid(i, j, k + 1) && indices(i, j, k + 1) >= 0)
                   ? A(i, j, k, i, j, k + 1) * precon[indices(i, j, k)] *
                         z[indices(i, j, k + 1)]
                   : 0.0);
          z[indices(i, j, k)] = t * precon[indices(i, j, k)];
        }
}

void pcg(MemoryBlock1Dd &x, FDMatrix3D &A, MemoryBlock1Dd &b,
         size_t maxNumberOfIterations, float tolerance) {
  // cpu memory
  MemoryBlock1Hd h_r(b.size(), 0);
  MemoryBlock1Hd h_z(b.size(), 0);
  MemoryBlock1Hd precon(b.size(), 0);
  FDMatrix3H h_A(A.gridSize());
  h_A.copy(A);
  mic0(precon, h_A, 0.97, 0.25);
  // device memory
  MemoryBlock1Dd m;
  MemoryBlock1Dd r(b.size(), 0); // residual
  MemoryBlock1Dd z(b.size(), 0); // auxiliar
  MemoryBlock1Dd s(b.size(), 0); // search
  std::cerr << "max " << maxNumberOfIterations << std::endl;
  // r = b - A * x
  mul(A, x, r);
  sub(b, r, r);
  if (infnorm(r, m) <= tolerance)
    return;
  // z = M * r
  memcpy(z, r);
  // memcpy(h_r, r);
  // memcpy(h_z, z);
  // applyMIC0(h_A, precon, h_r, h_z);
  // memcpy(z, h_z);
  // s = z
  memcpy(s, z);
  // sigma = z '* r
  double sigma = dot(z, r, m);
  // std::cerr << "sigma = " << sigma << std::endl;
  size_t it = 0;
  // std::cerr << "S " << s << std::endl;
  while (it < maxNumberOfIterations) {
    // z = As
    mul(A, s, z);
    // std::cerr << "Z " << z << std::endl;
    // std::cerr << "S " << s << std::endl;
    // alpha = sigma / (z '* s)
    double alpha = sigma / dot(z, s, m);
    // std::cerr << "alpha " << alpha << std::endl;
    // x = alpha * s + x
    axpy(alpha, s, x, x);
    // r = r - alpha * z
    axpy(-alpha, z, r, r);
    // std::cerr << "r norm test\n";
    // std::cerr << r << std::endl;
    if (infnorm(r, m) <= tolerance) {
      std::cerr << "PCG RUN with " << it << " iterations.\n";
      return;
    }
    // z = M * r
    memcpy(z, r);
    // memcpy(h_r, r);
    // memcpy(h_z, z);
    // applyMIC0(h_A, precon, h_r, h_z);
    // memcpy(z, h_z);
    // sigmaNew = z '* r
    double sigmaNew = dot(z, r, m);
    // std::cerr << "sigmaNew " << sigmaNew << std::endl;
    // if (sigmaNew < tolerance * tolerance)
    //   break;
    // s = z + (sigmaNew / sigma) * s
    axpy(sigmaNew / sigma, s, z, s);
    sigma = sigmaNew;
    ++it;
  }
  auto acc = h_A.accessor();
  std::cerr << "BAD PCG!\n" << acc << std::endl;
  std::cerr << b << std::endl;
  exit(-1);
}

} // namespace cuda

} // namespace poseidon
