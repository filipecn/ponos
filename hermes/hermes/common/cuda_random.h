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

#ifndef HERMES_COMMON_RANDOM_H
#define HERMES_COMMON_RANDOM_H

#include <hermes/common/cuda.h>

namespace hermes {

namespace cuda {

/** \brief Random Number Generator
 * Implements the "Halton Sequence".
 */
class HaltonSequence {
public:
  /// Default constructor.
  __host__ __device__ HaltonSequence() : base(2), ind(1) {}
  /// \param b base ( > 1)
  __host__ __device__ HaltonSequence(unsigned int b) : base(b), ind(1) {}
  /// \param b base ( > 1)
  __host__ __device__ void setBase(unsigned int b) {
    base = b;
    ind = 1;
  }
  __host__ __device__ void setIndex(unsigned int i) { ind = i; }
  /// pseudo-random floating-point number.
  /// \return a float in the range [0, 1)
  __host__ __device__ float randomFloat() {
    float result = 0.f;
    float f = 1.f;
    unsigned int i = ind++;
    while (i > 0) {
      f /= base;
      result += f * (i % base);
      i /= base;
    }
    return result;
  }

private:
  unsigned int base, ind;
};

} // namespace cuda

} // namespace hermes

#endif
