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

#include <poseidon/numeric/cuda_fd.h>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

template <typename T>
__global__ void __mul(FDMatrix2Accessor A, MemoryBlock1Accessor<T> x,
                      MemoryBlock1Accessor<T> b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!A.isIndexStored(i, j, i, j))
    return;
  int index = A.elementIndex(i, j);
  if (index < 0)
    return;
  b[index] = A(i, j, i, j) * x[index];
  int idx = A.elementIndex(i - 1, j);
  if (idx >= 0)
    b[index] += A(i, j, i - 1, j) * x[idx];
  idx = A.elementIndex(i + 1, j);
  if (idx >= 0)
    b[index] += A(i, j, i + 1, j) * x[idx];
  idx = A.elementIndex(i, j - 1);
  if (idx >= 0)
    b[index] += A(i, j, i, j - 1) * x[idx];
  idx = A.elementIndex(i, j + 1);
  if (idx >= 0)
    b[index] += A(i, j, i, j + 1) * x[idx];
}

template <> void mul(FDMatrix2D &A, MemoryBlock1Df &x, MemoryBlock1Df &b) {
  hermes::ThreadArrayDistributionInfo td(A.gridSize());
  __mul<<<td.gridSize, td.blockSize>>>(A.accessor(), x.accessor(),
                                       b.accessor());
}

template <> void mul(FDMatrix2D &A, MemoryBlock1Dd &x, MemoryBlock1Dd &b) {
  hermes::ThreadArrayDistributionInfo td(A.gridSize());
  __mul<<<td.gridSize, td.blockSize>>>(A.accessor(), x.accessor(),
                                       b.accessor());
}

template <typename T>
__global__ void __mul(FDMatrix3Accessor A, MemoryBlock1Accessor<T> x,
                      MemoryBlock1Accessor<T> b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (!A.isIndexStored(i, j, k, i, j, k))
    return;
  int index = A.elementIndex(i, j, k);
  if (index < 0)
    return;
  b[index] = A(i, j, k, i, j, k) * x[index];
  int idx = A.elementIndex(i - 1, j, k);
  if (idx >= 0)
    b[index] += A(i, j, k, i - 1, j, k) * x[idx];
  idx = A.elementIndex(i + 1, j, k);
  if (idx >= 0)
    b[index] += A(i, j, k, i + 1, j, k) * x[idx];
  idx = A.elementIndex(i, j - 1, k);
  if (idx >= 0)
    b[index] += A(i, j, k, i, j - 1, k) * x[idx];
  idx = A.elementIndex(i, j + 1, k);
  if (idx >= 0)
    b[index] += A(i, j, k, i, j + 1, k) * x[idx];
  idx = A.elementIndex(i, j, k - 1);
  if (idx >= 0)
    b[index] += A(i, j, k, i, j, k - 1) * x[idx];
  idx = A.elementIndex(i, j, k + 1);
  if (idx >= 0)
    b[index] += A(i, j, k, i, j, k + 1) * x[idx];
}

template <> void mul(FDMatrix3D &A, MemoryBlock1Df &x, MemoryBlock1Df &b) {
  hermes::ThreadArrayDistributionInfo td(A.gridSize());
  __mul<<<td.gridSize, td.blockSize>>>(A.accessor(), x.accessor(),
                                       b.accessor());
}

template <> void mul(FDMatrix3D &A, MemoryBlock1Dd &x, MemoryBlock1Dd &b) {
  hermes::ThreadArrayDistributionInfo td(A.gridSize());
  __mul<<<td.gridSize, td.blockSize>>>(A.accessor(), x.accessor(),
                                       b.accessor());
}

} // namespace cuda

} // namespace poseidon
