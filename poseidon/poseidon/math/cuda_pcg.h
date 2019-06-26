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

#ifndef POEIDON_MATH_CUDA_PCG_H
#define POEIDON_MATH_CUDA_PCG_H

#include <poseidon/math/cuda_fd.h>

namespace poseidon {

namespace cuda {

void mic0(hermes::cuda::MemoryBlock1Hd &precon, FDMatrix3H &A, double tal,
          double sigma);
// z = Mr
void applyMIC0(FDMatrix3H &h_A, hermes::cuda::MemoryBlock1Hd &precon,
               hermes::cuda::MemoryBlock1Hd &r,
               hermes::cuda::MemoryBlock1Hd &z);

void pcg(hermes::cuda::MemoryBlock1Dd &x, FDMatrix3D &A,
         hermes::cuda::MemoryBlock1Dd &rhs, size_t m, float tol);

int pcg(hermes::cuda::MemoryBlock1Dd &x, FDMatrix2D &A,
        hermes::cuda::MemoryBlock1Dd &rhs, size_t m, float tol);

} // namespace cuda

} // namespace poseidon

#endif