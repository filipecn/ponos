/*
 * Copyright (c) 2017 FilipeCN
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

#ifndef PONOS_GEOMETRY_NUMERIC_H
#define PONOS_GEOMETRY_NUMERIC_H

#include <ponos/common/defs.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace ponos {

#ifndef INFINITY
#define INFINITY FLT_MAX
#endif

class Constants {
public:
  static double pi;// = 3.14159265358979323846f;
};
///#ifndef PI
///#define PI 3.14159265358979323846f
///#endif
#ifndef PI_2
#define PI_2 6.28318530718
#endif
#ifndef INV_PI
#define INV_PI 0.31830988618379067154f
#endif
#ifndef INV_TWOPI
#define INV_TWOPI 0.15915494309189533577f
#endif
#ifndef INV_FOURPI
#define INV_FOURPI 0.07957747154594766788f
#endif

#ifndef SQR
#define SQR(A) ((A) * (A))
#endif

#ifndef CUBE
#define CUBE(A) ((A) * (A) * (A))
#endif

#ifndef TO_DEGREES
#define TO_DEGREES(A) ((A)*180.f / ponos::Constants::pi)
#endif

#ifndef TO_RADIANS
#define TO_RADIANS(A) ((A)*Constants::pi / 180.f)
#endif

#ifndef IS_ZERO
#define IS_ZERO(A) (fabs(A) < 1e-8)
#endif

#ifndef IS_EQUAL
#define IS_EQUAL(A, B) (fabs((A) - (B)) < 1e-6)
#endif

#ifndef IS_EQUAL_ERROR
#define IS_EQUAL_ERROR(A, B, C) (fabs((A) - (B)) < C)
#endif

#ifndef IS_BETWEEN
#define IS_BETWEEN(A, B, C) ((A) > (B) && (A) < (C))
#endif

#ifndef IS_BETWEEN_CLOSE
#define IS_BETWEEN_CLOSE(A, B, C) ((A) >= (B) && (A) <= (C))
#endif

/** \brief computes the arc-tangent of y/x
 * \param y
 * \param x
 * \returns angle (in radians) between **0** and **2PI**
 */
template <typename T> T atanPI_2(T y, T x) {
  T angle = std::atan2(y, x);
  if (angle < 0.0)
    return PI_2 + angle;
  return angle;
}
/** \brief normalization (dummy function)
 * \param n
 * \returns n
 */
inline float normalize(float n) { return n; }
/** \brief max function
 * \param a
 * \param b
 * \returns a if a >= b, b otherwise
 */
inline float max(float a, float b) { return std::max(a, b); }
/** \brief round
 * \param f **[in]**
 * \return ceil of **f**
 */
inline int ceil2Int(float f) { return static_cast<int>(f + 0.5f); }
/** \brief round
 * \param f **[in]**
 * \return floor of **f**
 */
inline int floor2Int(float f) { return static_cast<int>(f); }
/** \brief round
 * \param f **[in]**
 * \return next integer greater or equal to **f**
 */
inline int round2Int(float f) { return f + .5f; }
/** \brief modulus
 * \param a **[in]**
 * \param b **[in]**
 * \return the remainder of a / b
 */
inline float mod(int a, int b) {
  int n = a / b;
  a -= n * b;
  if (a < 0)
    a += b;
  return a;
}
template <typename T = float, typename S = float>
inline T bisect(T a, T b, S fa, S fb, S error, const std::function<S(T p)> &f) {
  if (fb < fa)
    return bisect(b, a, fb, fa, error, f);
  S zero = 0;
  if (fa > zero || IS_EQUAL_ERROR(fa, zero, error))
    return a;
  if (fb < zero || IS_EQUAL_ERROR(fb, zero, error))
    return b;
  T c = (a + b) / static_cast<T>(2);
  if (IS_EQUAL(a, c) || IS_EQUAL(b, c))
    return c;
  S fc = f(c);
  if (fc > zero)
    return bisect(a, c, fa, fc, error, f);
  return bisect(c, b, fc, fb, error, f);
}
/** \brief interpolation
 * \param t **[in]** parametric coordinate
 * \param a **[in]** lower bound **0**
 * \param b **[in]** upper bound **1**
 * \return linear interpolation between **a** and **b** at **t**.
 */
template <typename T = float, typename S = float>
inline S lerp(T t, const S &a, const S &b) {
  return (1.f - t) * a + t * b;
}
/** \brief bilinear interpolate
 * \param x **[in]** parametric coordinate in x
 * \param y **[in]** parametric coordinate in y
 * \param f00 **[in]** function value at **(0, 0)**
 * \param f10 **[in]** function value at **(1, 0)**
 * \param f11 **[in]** function value at **(1, 1)**
 * \param f01 **[in]** function value at **(0, 1)**
 * \return interpolated value at **(x,y)**
 */
template <typename T = float, typename S = float>
inline S bilerp(T x, T y, const S &f00, const S &f10, const S &f11,
                const S &f01) {
  return lerp(y, lerp(x, f00, f10), lerp(x, f01, f11));
}
template <typename T = float, typename S = float>
inline S trilerp(T tx, T ty, T tz, const S &f000, const S &f100, const S &f010,
                 const S &f110, const S &f001, const S &f101, const S &f011,
                 const S &f111) {
  return lerp(bilerp(f000, f100, f010, f110, tx, ty),
              bilerp(f001, f101, f011, f111, tx, ty), tz);
}
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
/** \brief interpolates to the nearest value
 * \param t **[in]** parametric coordinate
 * \param a **[in]** lower bound
 * \param b **[in]** upper bound
 * \return interpolation between **a** and **b** at **t**.
 */
template <typename T = float, typename S = float>
inline S nearest(T t, const S &a, const S &b) {
  return (t < static_cast<T>(0.5)) ? a : b;
}
/** \brief clamp
 * \param n **[in]** value
 * \param l **[in]** low
 * \param u **[in]** high
 * \return clame **b** to be in **[l, h]**
 */
template <typename T> T clamp(const T &n, const T &l, const T &u) {
  return std::max(l, std::min(n, u));
}
/** \brief smooth Hermit interpolation when **a** < **v** < **b**
 * \param v **[in]** coordinate
 * \param a **[in]** lower bound
 * \param b **[in]** upper bound
 * \return Hermit value between **0** and **1**
 */
template<typename T>
T smoothStep(T v, T a, T b) {
  float t = clamp((v - a) / (b - a), T(0), T(1));
  return t * t * (T(3) - 2 * t);
}
/** \brief linear interpolation
 * \param v **[in]** coordinate
 * \param a **[in]** lower bound
 * \param b **[in]** upper bound
 * \return linear value between **0** and **1**
 */
inline float linearStep(float v, float a, float b) {
  return clamp((v - a) / (b - a), 0.f, 1.f);
}
/** \brief log
 * \param x **[in]** value
 * \return base-2 logarithm of **x**
 */
inline float log2(float x) {
  static float invLog2 = 1.f / logf(2.f);
  return logf(x) * invLog2;
}
/** \brief log
 * \param x **[in]** value
 * \return integer base-2 logarithm of **x**
 */
inline int log2Int(float x) { return floor2Int(log2(x)); }
/** \brief query
 * \param v **[in]** value
 * \return **true** if **v** is power of 2
 */
inline bool isPowerOf2(int v) { return (v & (v - 1)) == 0; }
/** \brief round
 * \param v **[in]** value
 * \return the next number in power of 2
 */
inline uint32 roundUpPow2(uint32 v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}
inline bool solve_quadratic(float A, float B, float C, float &t0, float &t1) {
  float delta = B * B - 4.f * A * C;
  if (IS_ZERO(A) || delta <= 0.)
    return false;
  float sDelta = sqrtf(delta);
  float q;
  if (B < 0.)
    q = -0.5 * (B - sDelta);
  else
    q = -0.5f * (B + sDelta);
  t0 = q / A;
  t1 = C / q;
  if (t0 > t1)
    std::swap(t0, t1);
  return true;
}
inline float trilinear_hat_function(float r) {
  if (0.f <= r && r <= 1.f)
    return 1.f - r;
  if (-1.f <= r && r <= 0.f)
    return 1.f + r;
  return 0.f;
}
inline float quadraticBSpline(float r) {
  if (-1.5f <= r && r < -0.5f)
    return 0.5f * (r + 1.5f) * (r + 1.5f);
  if (-0.5f <= r && r < 0.5f)
    return 0.75f - r * r;
  if (0.5f <= r && r < 1.5f)
    return 0.5f * (1.5f - r) * (1.5f - r);
  return 0.0f;
}
template <typename T>
inline T bilinearInterpolation(T f00, T f10, T f11, T f01, T x, T y) {
  return f00 * (1.0 - x) * (1.0 - y) + f10 * x * (1.0 - y) +
         f01 * (1.0 - x) * y + f11 * x * y;
}
template <typename T> inline T cubicInterpolate(T p[4], T x) {
  return p[1] +
         0.5 * x * (p[2] - p[0] +
                    x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
                         x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}
template <typename T> inline T bicubicInterpolate(T p[4][4], T x, T y) {
  T arr[4];
  arr[0] = cubicInterpolate(p[0], y);
  arr[1] = cubicInterpolate(p[1], y);
  arr[2] = cubicInterpolate(p[2], y);
  arr[3] = cubicInterpolate(p[3], y);
  return cubicInterpolate(arr, x);
}
template <typename T>
inline T trilinearInterpolate(float *p, T ***data, T b,
                              const int dimensions[3]) {
  int i0 = p[0], j0 = p[1], k0 = p[2];
  int i1 = p[0] + 1, j1 = p[1] + 1, k1 = p[2] + 1;
  float x = p[0] - i0;
  float y = p[1] - j0;
  float z = p[2] - k0;
  T v000 = (i0 < 0 || j0 < 0 || k0 < 0 || i0 >= dimensions[0] ||
            j0 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i0][j0][k0];
  T v001 = (i0 < 0 || j0 < 0 || k1 < 0 || i0 >= dimensions[0] ||
            j0 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i0][j0][k1];
  T v010 = (i0 < 0 || j1 < 0 || k0 < 0 || i0 >= dimensions[0] ||
            j1 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i0][j1][k0];
  T v011 = (i0 < 0 || j1 < 0 || k1 < 0 || i0 >= dimensions[0] ||
            j1 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i0][j1][k1];
  T v100 = (i1 < 0 || j0 < 0 || k0 < 0 || i1 >= dimensions[0] ||
            j0 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i1][j0][k0];
  T v101 = (i1 < 0 || j0 < 0 || k1 < 0 || i1 >= dimensions[0] ||
            j0 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i1][j0][k1];
  T v110 = (i1 < 0 || j1 < 0 || k0 < 0 || i1 >= dimensions[0] ||
            j1 >= dimensions[1] || k0 >= dimensions[2])
               ? b
               : data[i1][j1][k0];
  T v111 = (i1 < 0 || j1 < 0 || k1 < 0 || i1 >= dimensions[0] ||
            j1 >= dimensions[1] || k1 >= dimensions[2])
               ? b
               : data[i1][j1][k1];
  return v000 * (1.f - x) * (1.f - y) * (1.f - z) +
         v100 * x * (1.f - y) * (1.f - z) + v010 * (1.f - x) * y * (1.f - z) +
         v110 * x * y * (1.f - z) + v001 * (1.f - x) * (1.f - y) * z +
         v101 * x * (1.f - y) * z + v011 * (1.f - x) * y * z + v111 * x * y * z;
}
template <typename T> inline T tricubicInterpolate(float *p, T ***data) {
  int x, y, z;
  int i, j, k;
  float dx, dy, dz;
  float u[4], v[4], w[4];
  T r[4], q[4];
  T vox = T(0);

  x = (int)p[0], y = (int)p[1], z = (int)p[2];
  dx = p[0] - (float)x, dy = p[1] - (float)y, dz = p[2] - (float)z;

  u[0] = -0.5 * CUBE(dx) + SQR(dx) - 0.5 * dx;
  u[1] = 1.5 * CUBE(dx) - 2.5 * SQR(dx) + 1;
  u[2] = -1.5 * CUBE(dx) + 2 * SQR(dx) + 0.5 * dx;
  u[3] = 0.5 * CUBE(dx) - 0.5 * SQR(dx);

  v[0] = -0.5 * CUBE(dy) + SQR(dy) - 0.5 * dy;
  v[1] = 1.5 * CUBE(dy) - 2.5 * SQR(dy) + 1;
  v[2] = -1.5 * CUBE(dy) + 2 * SQR(dy) + 0.5 * dy;
  v[3] = 0.5 * CUBE(dy) - 0.5 * SQR(dy);

  w[0] = -0.5 * CUBE(dz) + SQR(dz) - 0.5 * dz;
  w[1] = 1.5 * CUBE(dz) - 2.5 * SQR(dz) + 1;
  w[2] = -1.5 * CUBE(dz) + 2 * SQR(dz) + 0.5 * dz;
  w[3] = 0.5 * CUBE(dz) - 0.5 * SQR(dz);

  int ijk[3] = {x - 1, y - 1, z - 1};
  for (k = 0; k < 4; k++) {
    q[k] = 0;
    for (j = 0; j < 4; j++) {
      r[j] = 0;
      for (i = 0; i < 4; i++) {
        r[j] += u[i] * data[ijk[0]][ijk[1]][ijk[2]];
        ijk[0]++;
      }
      q[k] += v[j] * r[j];
      ijk[0] = x - 1;
      ijk[1]++;
    }
    vox += w[k] * q[k];
    ijk[0] = x - 1;
    ijk[1] = y - 1;
    ijk[2]++;
  }
  return (vox < T(0) ? T(0.0) : vox);
}
template <typename T>
inline T tricubicInterpolate(float *p, T ***data, T b,
                             const int dimensions[3]) {
  int x, y, z;
  int i, j, k;
  float dx, dy, dz;
  float u[4], v[4], w[4];
  T r[4], q[4];
  T vox = T(0);

  x = (int)p[0], y = (int)p[1], z = (int)p[2];
  dx = p[0] - (float)x, dy = p[1] - (float)y, dz = p[2] - (float)z;

  u[0] = -0.5 * CUBE(dx) + SQR(dx) - 0.5 * dx;
  u[1] = 1.5 * CUBE(dx) - 2.5 * SQR(dx) + 1;
  u[2] = -1.5 * CUBE(dx) + 2 * SQR(dx) + 0.5 * dx;
  u[3] = 0.5 * CUBE(dx) - 0.5 * SQR(dx);

  v[0] = -0.5 * CUBE(dy) + SQR(dy) - 0.5 * dy;
  v[1] = 1.5 * CUBE(dy) - 2.5 * SQR(dy) + 1;
  v[2] = -1.5 * CUBE(dy) + 2 * SQR(dy) + 0.5 * dy;
  v[3] = 0.5 * CUBE(dy) - 0.5 * SQR(dy);

  w[0] = -0.5 * CUBE(dz) + SQR(dz) - 0.5 * dz;
  w[1] = 1.5 * CUBE(dz) - 2.5 * SQR(dz) + 1;
  w[2] = -1.5 * CUBE(dz) + 2 * SQR(dz) + 0.5 * dz;
  w[3] = 0.5 * CUBE(dz) - 0.5 * SQR(dz);

  int ijk[3] = {x - 1, y - 1, z - 1};
  for (k = 0; k < 4; k++) {
    q[k] = 0;
    for (j = 0; j < 4; j++) {
      r[j] = 0;
      for (i = 0; i < 4; i++) {
        if (ijk[0] < 0 || ijk[0] >= dimensions[0] || ijk[1] < 0 ||
            ijk[1] >= dimensions[1] || ijk[2] < 0 || ijk[2] >= dimensions[2])
          r[j] += u[i] * b;
        else
          r[j] += u[i] * data[ijk[0]][ijk[1]][ijk[2]];
        ijk[0]++;
      }
      q[k] += v[j] * r[j];
      ijk[0] = x - 1;
      ijk[1]++;
    }
    vox += w[k] * q[k];
    ijk[0] = x - 1;
    ijk[1] = y - 1;
    ijk[2]++;
  }
  return (vox < T(0) ? T(0.0) : vox);
}
inline float smooth(float a, float b) {
  return std::max(0.f, 1.f - a / SQR(b));
}
inline float sharpen(const float &r2, const float &h) {
  return std::max(h * h / std::max(r2, static_cast<float>(1.0e-5)) - 1.0f,
                  0.0f);
}
/** \brief split
 * \param n **[in]** number
 * \return the number with the bits of **n** splitted and separated by 1 bit.
 * Ex: n = 1101, return 101001
 */
inline uint32 separateBy1(uint32 n) {
  n = (n ^ (n << 8)) & 0x00ff00ff;
  n = (n ^ (n << 4)) & 0x0f0f0f0f;
  n = (n ^ (n << 2)) & 0x33333333;
  n = (n ^ (n << 1)) & 0x55555555;
  return n;
}
/** \brief morton code
 * \param x **[in]** number
 * \param y **[in]** number
 * \return the morton code of **x** and **y**. The morton code is the
 * combination of the two binary numbers with the bits interleaved.
 */
inline uint32 mortonCode(uint32 x, uint32 y) {
  return (separateBy1(y) << 1) + separateBy1(x);
}

inline uint32_t separateBy2(uint32_t n) {
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;
  return n;
}

/** \brief morton code
 * \param x **[in]** number
 * \param y **[in]** number
 * \param z **[in]** number
 * \return the morton code of **x**, **y** and **z**. The morton code is the
 * combination of the two binary numbers with the bits interleaved.
 */
inline uint32_t mortonCode(uint32_t x, uint32_t y, uint32_t z) {
  return (separateBy2(z) << 2) + (separateBy2(y) << 1) + separateBy2(x);
}

} // ponos namespace

#endif
