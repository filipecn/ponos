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

#ifndef POSEIDON_SIMULATION_CUDA_DEFS_H
#define POSEIDON_SIMULATION_CUDA_DEFS_H

#include <hermes/numeric/cuda_grid.h>

namespace poseidon {

namespace cuda {

enum class MaterialType { FLUID = 0, AIR, SOLID };

inline std::ostream &operator<<(std::ostream &os, MaterialType m) {
  if (m == MaterialType::FLUID)
    os << 'F';
  else if (m == MaterialType::AIR)
    os << 'A';
  else
    os << 'S';
  return os;
}

using RegularGrid2Dm =
    hermes::cuda::RegularGrid2<hermes::cuda::MemoryLocation::DEVICE,
                               MaterialType>;
using RegularGrid2Hm =
    hermes::cuda::RegularGrid2<hermes::cuda::MemoryLocation::HOST,
                               MaterialType>;
} // namespace cuda

} // namespace poseidon

#endif