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

enum class AddressMode { /*REPEAT,*/ CLAMP_TO_EDGE,
                         BORDER,
                         WRAP,
                         MIRROR,
                         NONE };

enum class FilterMode { LINEAR, POINT };

namespace cuda {

class Constants {
public:
  static double pi;     // = 3.14159265358979323846f;
  static double two_pi; // = 6.28318530718
  template <typename T> __host__ __device__ static T lowest() {
    return 0xfff0000000000000;
  }
  template <typename T> __host__ __device__ static T greatest() {
    return 0x7ff0000000000000;
  }
};

class Check {
public:
  template <typename T> __host__ __device__ static bool isEqual(T a, T b) {
    return fabs(a - b) < 1e-8;
  }
};

} // namespace cuda

} // namespace hermes

#endif
