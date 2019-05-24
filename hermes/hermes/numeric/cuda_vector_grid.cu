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

#include <hermes/numeric/cuda_vector_field.h>

namespace hermes {

namespace cuda {

// __global__ void __fill3(VectorGrid3Accessor acc, bbox3f region, vec3f value,
//                         bool overwrite) {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   int z = blockIdx.z * blockDim.z + threadIdx.z;
//   if (acc.isIndexStored(x, y, z)) {
//     auto wp = acc.worldPosition(x, y, z);
//     if (region.contains(wp)) {
//       if (overwrite)
//         acc(x, y, z) = value;
//       else
//         acc(x, y, z) += value;
//     }
//   }
// }

// template <>
// void fill3(VectorGrid3D &grid, const bbox3f &region, vec3f value,
//            bool overwrite) {
//   ThreadArrayDistributionInfo td(grid.resolution());
//   __fill3<<<td.gridSize, td.blockSize>>>(grid.accessor(), region, value,
//                                          overwrite);
// }

} // namespace cuda
} // namespace hermes
