/// Copyright (c) 2017, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file numeric.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2017-03-14
///
///\brief

#ifndef PONOS_GEOMETRY_NUMERIC_H
#define PONOS_GEOMETRY_NUMERIC_H

#include <ponos/common/defs.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace ponos {

#ifndef INFINITY
#define INFINITY FLT_MAX
#endif
///\brief
///
struct Constants {
  static constexpr real_t pi = 3.14159265358979323846;
  static constexpr real_t two_pi = 6.28318530718;
  static constexpr real_t inv_pi = 0.31830988618379067154;
  static constexpr real_t inv_two_pi = 0.15915494309189533577;
  static constexpr real_t inv_four_pi = 0.07957747154594766788;
  static real_t real_infinity;
  template<typename T> static constexpr T lowest() {
    return std::numeric_limits<T>::lowest();
  }
  template<typename T> static constexpr T greatest() {
    return std::numeric_limits<T>::max();
  }
};
///\brief
///
struct Check {
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_zero(T a) {
    return std::fabs(a) < 1e-8;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_equal(T a, T b) {
    return fabs(a - b) < 1e-8;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\param e **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_equal(T a, T b, T e) {
    return fabs(a - b) < e;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between(T x, T a, T b) {
    return x > a && x < b;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between_closed(T x, T a, T b) {
    return x >= a && x <= b;
  }
};

template<typename T> constexpr T SQR(T a) { return a * a; }
template<typename T> constexpr T CUBE(T a) { return a * a * a; }
template<typename T> constexpr T DEGREES(T a) {
  return a * 180.f / Constants::pi;
}
template<typename T> constexpr T RADIANS(T a) {
  return a * Constants::pi / 180.f;
}

template<typename T> char sign(T n) {
  if (n < 0)
    return -1;
  if (n > 0)
    return 1;
  return 0;
}

/** \brief computes the arc-tangent of y/x
 * \param y
 * \param x
 * \returns angle (in radians) between **0** and **2PI**
 */
template<typename T> T atanPI_2(T y, T x) {
  T angle = std::atan2(y, x);
  if (angle < 0.0)
    return Constants::two_pi + angle;
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
template<typename T = float, typename S = float>
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
/** \brief clamp
 * \param n **[in]** value
 * \param l **[in]** low
 * \param u **[in]** high
 * \return clamp **b** to be in **[l, h]**
 */
template<typename T> T clamp(T n, T l, T u) {
  return std::max(l, std::min(n, u));
}
/** \brief smooth Hermit interpolation when **a** < **v** < **b**
 * \param v **[in]** coordinate
 * \param a **[in]** lower bound
 * \param b **[in]** upper bound
 * \return Hermit value between **0** and **1**
 */
template<typename T> T smoothStep(T v, T a, T b) {
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
inline u32 roundUpPow2(u32 v) {
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
  if (Check::is_zero(A) || delta <= 0.)
    return false;
  float sDelta = sqrtf(delta);
  float q;
  if (B < 0.)
    q = -0.5f * (B - sDelta);
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
inline u32 separateBy1(u32 n) {
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
inline u32 encodeMortonCode(u32 x, u32 y) {
  return (separateBy1(y) << 1) + separateBy1(x);
}

inline u32 separateBy2(u32 n) {
  n = (n ^ (n << 16u)) & 0xff0000ff;
  n = (n ^ (n << 8u)) & 0x0300f00f;
  n = (n ^ (n << 4u)) & 0x030c30c3;
  n = (n ^ (n << 2u)) & 0x09249249;
  return n;
}

/** \brief morton code
 * \param x **[in]** number
 * \param y **[in]** number
 * \param z **[in]** number
 * \return the morton code of **x**, **y** and **z**. The morton code is the
 * combination of the two binary numbers with the bits interleaved.
 */
inline u32 encodeMortonCode(u32 x, u32 y, u32 z) {
  return (separateBy2(z) << 2) + (separateBy2(y) << 1) + separateBy2(x);
}

template<typename T> T dot(const std::vector<T> &a, const std::vector<T> &b) {
  ASSERT_FATAL(a.size() == b.size());
  T sum = T(0);
  for (size_t i = 0; i < a.size(); i++)
    sum = sum + a[i] * b[i];
  return sum;
}

} // namespace ponos

#endif
