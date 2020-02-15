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

#ifndef HERMES_GEOMETRY_CUDA_NUMERIC_H
#define HERMES_GEOMETRY_CUDA_NUMERIC_H

#include <hermes/common/cuda.h>
#include <hermes/geometry/cuda_point.h>

namespace hermes {

namespace cuda {

template <typename T> __host__ __device__ T min(const T &a, const T &b) {
  if (a < b)
    return a;
  return b;
}

template <typename T> __host__ __device__ T max(const T &a, const T &b) {
  if (a > b)
    return a;
  return b;
}

template <typename T> __host__ __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T> __host__ __device__ int sign(T a) {
  return a >= 0 ? 1 : -1;
}

/// \param t **[in]** parametric coordinate
/// \param a **[in]** lower bound **0**
/// \param b **[in]** upper bound **1**
/// \return linear interpolation between **a** and **b** at **t**.
template <typename T = float, typename S = float>
inline __host__ __device__ S lerp(T t, const S &a, const S &b) {
  return (1.f - t) * a + t * b;
}
/// \param x **[in]** parametric coordinate in x
/// \param y **[in]** parametric coordinate in y
/// \param f00 **[in]** function value at **(0, 0)**
/// \param f10 **[in]** function value at **(1, 0)**
/// \param f11 **[in]** function value at **(1, 1)**
/// \param f01 **[in]** function value at **(0, 1)**
/// \return interpolated value at **(x,y)**
template <typename T = float, typename S = float>
inline __host__ __device__ S bilerp(T x, T y, const S &f00, const S &f10,
                                    const S &f11, const S &f01) {
  return lerp(y, lerp(x, f00, f10), lerp(x, f01, f11));
}
/// \tparam float
/// \tparam float
/// \param tx
/// \param ty
/// \param tz
/// \param f000
/// \param f100
/// \param f010
/// \param f110
/// \param f001
/// \param f101
/// \param f011
/// \param f111
/// \return S
template <typename T = float, typename S = float>
inline S trilerp(T tx, T ty, T tz, const S &f000, const S &f100, const S &f010,
                 const S &f110, const S &f001, const S &f101, const S &f011,
                 const S &f111) {
  return lerp(bilerp(f000, f100, f010, f110, tx, ty),
              bilerp(f001, f101, f011, f111, tx, ty), tz);
}
/// \tparam S
/// \tparam T
/// \param f0
/// \param f1
/// \param f2
/// \param f3
/// \param f
/// \return S
template <typename S, typename T>
inline S catmullRomSpline(const S &f0, const S &f1, const S &f2, const S &f3,
                          T f) {
  S d1 = (f2 - f0) / 2;
  S d2 = (f3 - f1) / 2;
  S D1 = f2 - f1;

  S a3 = d1 + d2 - 2 * D1;
  S a2 = 3 * D1 - 2 * d1 - d2;
  S a1 = d1;
  S a0 = f1;

  return a3 * CUBE(f) + a2 * SQR(f) + a1 * f + a0;
}
/// interpolates to the nearest value
/// \param t **[in]** parametric coordinate
/// \param a **[in]** lower bound
/// \param b **[in]** upper bound
/// \return interpolation between **a** and **b** at **t**.
template <typename T = float, typename S = float>
inline S nearest(T t, const S &a, const S &b) {
  return (t < static_cast<T>(0.5)) ? a : b;
}
/// \param n **[in]** value
/// \param l **[in]** low
/// \param u **[in]** high
/// \return clamp **b** to be in **[l, h]**
template <typename T>
__host__ __device__ T clamp(const T &n, const T &l, const T &u) {
  return fmaxf(l, fminf(n, u));
}
///  smooth Hermit interpolation when **a** < **v** < **b**
/// \param v **[in]** coordinate
/// \param a **[in]** lower bound
/// \param b **[in]** upper bound
/// \return Hermit value between **0** and **1**
template <typename T> T smoothStep(T v, T a, T b) {
  float t = clamp((v - a) / (b - a), T(0), T(1));
  return t * t * (T(3) - 2 * t);
}
/// \param v **[in]** coordinate
/// \param a **[in]** lower bound
/// \param b **[in]** upper bound
/// \return linear value between **0** and **1**
inline float linearStep(float v, float a, float b) {
  return clamp((v - a) / (b - a), 0.f, 1.f);
}
/// \param x **[in]** value
/// \return base-2 logarithm of **x**
inline float log2(float x) {
  static float invLog2 = 1.f / logf(2.f);
  return logf(x) * invLog2;
}
/// \param v **[in]** value
/// \return **true** if **v** is power of 2
inline bool isPowerOf2(int v) { return (v & (v - 1)) == 0; }
/// \param a
/// \param b
/// \return float
inline float smooth(float a, float b) { return fmaxf(0.f, 1.f - a / (b * b)); }
/// \param r2
/// \param h
/// \return float
inline float sharpen(const float &r2, const float &h) {
  return fmaxf(h * h / fmaxf(r2, static_cast<float>(1.0e-5)) - 1.0f, 0.0f);
}

__device__ __host__ inline unsigned int separateBitsBy1(unsigned int n) {
  n = (n ^ (n << 8)) & 0x00ff00ff;
  n = (n ^ (n << 4)) & 0x0f0f0f0f;
  n = (n ^ (n << 2)) & 0x33333333;
  n = (n ^ (n << 1)) & 0x55555555;
  return n;
}

__device__ __host__ inline unsigned int separateBitsBy2(unsigned int n) {
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;
  return n;
}

__device__ __host__ inline unsigned int
interleaveBits(unsigned int x, unsigned int y, unsigned int z) {
  return (separateBitsBy2(z) << 2) + (separateBitsBy2(y) << 1) +
         separateBitsBy2(x);
}

__device__ __host__ inline unsigned int interleaveBits(unsigned int x,
                                                       unsigned int y) {
  return (separateBitsBy1(y) << 1) + separateBitsBy1(x);
}

template <typename T>
__device__ __host__ unsigned int mortonCode(const Point3<T> &v) {
  return interleaveBits(v[0], v[1], v[2]);
}

template <typename T>
__device__ __host__ unsigned int mortonCode(const Point2<T> &v) {
  return interleaveBits(v[0], v[1]);
}

} // namespace cuda

} // namespace hermes

#endif
