/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_COMMON_PARALLEL_H
#define HERMES_COMMON_PARALLEL_H

#include <hermes/common/size.h>
#include <hermes/geometry/vector.h>

namespace hermes {

namespace cuda {

#define GPU_BLOCK_SIZE 1024

struct ThreadArrayDistributionInfo {
  __host__ __device__ ThreadArrayDistributionInfo(u32 n) {
    blockSize.x = GPU_BLOCK_SIZE;
    gridSize.x = n / blockSize.x + 1;
  }
  __host__ __device__ ThreadArrayDistributionInfo(size2 s) {
    blockSize = dim3(16, 16);
    gridSize = dim3((s.width + blockSize.x - 1) / blockSize.x,
                    (s.height + blockSize.y - 1) / blockSize.y);
  }
  __host__ __device__ ThreadArrayDistributionInfo(unsigned int w,
                                                  unsigned int h) {
    blockSize = dim3(16, 16);
    gridSize = dim3((w + blockSize.x - 1) / blockSize.x,
                    (h + blockSize.y - 1) / blockSize.y);
  }
  __host__ __device__ ThreadArrayDistributionInfo(cuda::vec2u resolution) {
    blockSize = dim3(16, 16);
    gridSize = dim3((resolution.x + blockSize.x - 1) / blockSize.x,
                    (resolution.y + blockSize.y - 1) / blockSize.y);
  }
  __host__ __device__ ThreadArrayDistributionInfo(cuda::vec3u resolution) {
    blockSize = dim3(16, 16);
    gridSize = dim3((resolution.x + blockSize.x - 1) / blockSize.x,
                    (resolution.y + blockSize.y - 1) / blockSize.y,
                    (resolution.z + blockSize.z - 1) / blockSize.z);
  }
  __host__ __device__ ThreadArrayDistributionInfo(unsigned int w,
                                                  unsigned int h,
                                                  unsigned int d) {
    blockSize = dim3(8, 8, 8);
    gridSize = dim3((w + blockSize.x - 1) / blockSize.x,
                    (h + blockSize.y - 1) / blockSize.y,
                    (d + blockSize.z - 1) / blockSize.z);
  }
  dim3 gridSize;
  dim3 blockSize;
};

} // namespace cuda

} // namespace hermes

#endif
