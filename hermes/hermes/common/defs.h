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

#ifndef HERMES_COMMON_DEFS_H
#define HERMES_COMMON_DEFS_H

namespace hermes {

namespace cuda {

enum class MemoryLocation { DEVICE, HOST };

class Constants {
public:
  __host__ __device__ static double pi() { return 3.14159265358979323846; }
  __host__ __device__ static double two_pi() { return 6.28318530718; }
  template <typename T> __host__ __device__ static T lowest() {
    return 0xfff0000000000000;
  }
  template <typename T> __host__ __device__ static T greatest() {
    return 0x7ff0000000000000;
  }
  __host__ __device__ static int lowest_int() { return -2147483647; }
  __host__ __device__ static int greatest_int() { return 2147483647; }
};

class Check {
public:
  template <typename T> __host__ __device__ static bool is_equal(T a, T b) {
    return fabsf(a - b) < 1e-8f;
  }
};

} // namespace cuda

} // namespace hermes

#endif
