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

#ifndef POSEIDON_NUMERIC_CUDA_NUMERIC_H
#define POSEIDON_NUMERIC_CUDA_NUMERIC_H

#include <hermes/numeric/cuda_interpolation.h>
#include <hermes/storage/cuda_storage_utils.h>
#include <poseidon/simulation/cuda_integrator.h>

namespace poseidon {

namespace cuda {

/// Computes first order accurate forward difference approximation for first
/// derivative in midway between (i, j) and (i+1, j) or (i, j+1)
/// \param phi field accessor reference
/// \param i starting i coordinate
/// \param j starting j coordinate
/// \param dim differentiation axis (0 = x, 1 = y)
/// \return float (D0_plus_one - D0) / dx
inline __host__ __device__ float
D1_plus_half(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i, int j,
             size_t dim) {
  // printf("D1ph %d %d %f  %d %d %f\n", i + (0 | !dim), j + (1 & dim),
  //  phi(i + (0 | !dim), j + (1 & dim)), i, j, phi(i, j));
  return (phi(i + (0 | !dim), j + (1 & dim)) - phi(i, j)) / phi.spacing()[dim];
}
/// Computes first order accurate backward difference approximation for first
/// derivative in midway between (i-1, j) or (i, j-1) and (i, j)
/// \param phi field accessor reference
/// \param i starting i coordinate
/// \param j starting j coordinate
/// \param dim differentiation axis (0 = x, 1 = y)
/// \return float (D0 - D0_minus_one) / dx
inline __host__ __device__ float
D1_minus_half(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i,
              int j, size_t dim) {
  return (phi(i, j) - phi(i - (0 | !dim), j - (1 & dim))) / phi.spacing()[dim];
}
/// Computes second order approximation for first derivative at (i,j)
/// \param phi field accessor reference
/// \param i starting i coordinate
/// \param j starting j coordinate
/// \param dim differentiation axis (0 = x, 1 = y)
/// \return float (D1_plus_half - D1_minus_half) / 2dx
inline __host__ __device__ float
D2(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i, int j,
   size_t dim) {
  return (D1_plus_half(phi, i, j, dim) - D1_minus_half(phi, i, j, dim)) /
         (2 * phi.spacing()[dim]);
}
/// Computes third order accurate forward difference approximation for first
/// derivative in midway between (i, j) and (i+1, j) or (i, j+1)
/// \param phi field accessor reference
/// \param i starting i coordinate
/// \param j starting j coordinate
/// \param dim differentiation axis (0 = x, 1 = y)
/// \return float (D2_plus_one - D2) / 3dx
inline __host__ __device__ float
D3_plus_half(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i, int j,
             size_t dim) {
  return (D2(phi, i + (0 | !dim), j + (1 & dim), dim) - D2(phi, i, j, dim)) /
         (3 * phi.spacing()[dim]);
}
/// Computes third order accurate forward difference approximation for first
/// derivative in midway between (i-1, j) or (i, j-1) and (i, j)
/// \param phi field accessor reference
/// \param i starting i coordinate
/// \param j starting j coordinate
/// \param dim differentiation axis (0 = x, 1 = y)
/// \return float (D2 - D2_minus_one) / 3dx
inline __host__ __device__ float
D3_minus_half(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i,
              int j, size_t dim) {
  return (D2(phi, i, j, dim) - D2(phi, i - (0 | !dim), j - (1 & dim), dim)) /
         (3 * phi.spacing()[dim]);
}

/// \param phi **[in]**
/// \param i **[in]**
/// \param j **[in]**
/// \param accuracy_order **[in]**
/// \return hermes::cuda::vec2f
inline __host__ __device__ hermes::cuda::vec2f
gradientAt(const hermes::cuda::RegularGrid2Accessor<float> &phi, int i, int j,
           size_t accuracy_order = 1,
           const hermes::cuda::vec2f &v = hermes::cuda::vec2f()) {
  using namespace hermes::cuda;
  vec2f dphi;
  // printf("gradient at %d, %d\n", i, j);
  // compute each phi derivative
  for (size_t dim = 0; dim < 2; dim++) {
    // construct approximation polynom for phi first derivative
    // Dphi = dq1 + dq2 + dq3
    // compute starting index k based on upwind
    vec2i k(i, j);
    k[dim] = v[dim] > 0 ? k[dim] - 1 : k[dim];
    // printf("for dim %d : k is %d %d\n", dim, k.x, k.y);
    // dq1 = D1_plus_half
    float dq1 = D1_plus_half(phi, k.x, k.y, dim);
    // printf("= %f\n", dq1);
    dphi[dim] = dq1;
    if (accuracy_order < 2)
      continue;
    // compare |D2| and |D2_plus_one| to avoid big variations
    float D2k = D2(phi, k.x, k.y, dim);
    float D2k_plus_one = D2(phi, k.x + (0 | !dim), k.y + (1 & dim), dim);
    vec2i k_star = k;
    float c = D2k_plus_one;
    if (fabsf(D2k) <= fabsf(D2k_plus_one)) {
      k_star[dim] -= 1;
      c = D2k;
    }
    // dq2 = c(2(i-k)-1)dx
    vec2i ij(i, j);
    float dq2 = c * (2 * (ij[dim] - k[dim]) - 1) * phi.spacing()[dim];
    dphi[dim] += dq2;
    if (accuracy_order < 3)
      continue;
    // compare |D3k_star_plus_half| and |D3k_star_plus_3/2| to avoid big
    // variations
    float D3k_star_plus_half = D3_plus_half(phi, k_star.x, k_star.y, dim);
    float D3k_star_plus_3_2 =
        D3_plus_half(phi, k_star.x + (0 | !dim), k_star.y + (1 & dim), dim);
    c = D3k_star_plus_3_2;
    if (fabsf(D3k_star_plus_half) <= fabsf(D3k_star_plus_3_2))
      c = D3k_star_plus_half;
    // dq3 = c(3(i-k_star)^2 - 6(i-k_star) + 2)(dx)^2
    float imks = ij[dim] - k_star[dim];
    float dq3 = c * (3 * imks * imks * imks - 6 * imks + 2) *
                phi.spacing()[dim] * phi.spacing()[dim];
    dphi[dim] += dq3;
  }
  return dphi;
}

/// \param phi **[in]**
/// \param i **[in]**
/// \param j **[in]**
/// \param accuracy_order **[in]**
/// \return hermes::cuda::vec2f
inline __host__ __device__ hermes::cuda::vec3f
gradientAt(const hermes::cuda::RegularGrid3Accessor<float> &phi, int i, int j,
           int K, size_t accuracy_order = 1,
           const hermes::cuda::vec3f &v = hermes::cuda::vec3f()) {
  using namespace hermes::cuda;
  vec3f dphi;

  return dphi;
}

} // namespace cuda

} // namespace poseidon

#endif