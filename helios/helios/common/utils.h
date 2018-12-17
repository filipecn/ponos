/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef HELIOS_COMMON_UTILS_H
#define HELIOS_COMMON_UTILS_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace helios {

/// Interprets a floating-point value into a integer type
/// \param f float value
/// \return  a 32 bit unsigned integer containing the bits of **f**
inline uint32_t floatToBits(float f) {
  uint32_t ui(0);
  memcpy(&ui, &f, sizeof(float));
  return ui;
}

/// Fills a float variable data
/// \param ui bits
/// \return a float built from bits of **ui**
inline float bitsToFloat(uint32_t ui) {
  float f(0.f);
  memcpy(&f, &ui, sizeof(uint32_t));
  return f;
}

/// Interprets a doubleing-point value into a integer type
/// \param d double value
/// \return  a 64 bit unsigned integer containing the bits of **f**
inline uint64_t doubleToBits(double d) {
  uint64_t ui(0);
  memcpy(&ui, &d, sizeof(double));
  return ui;
}

/// Fills a double variable data
/// \param ui bits
/// \return a double built from bits of **ui**
inline double bitsToDouble(uint64_t ui) {
  double d(0.f);
  memcpy(&d, &ui, sizeof(uint64_t));
  return d;
}

/// Computes the next greater representable floating-point value
/// \param v floating point value
/// \return the next greater floating point value
inline float nextFloatUp(float v) {
  if (std::isinf(v) && v > 0.)
    return v;
  if (v == -0.f)
    v = 0.f;
  uint32_t ui = floatToBits(v);
  if (v >= 0)
    ++ui;
  else
    --ui;
  return bitsToFloat(ui);
}

/// Computes the next smaller representable floating-point value
/// \param v floating point value
/// \return the next smaller floating point value
inline float nextFloatDown(float v) {
  if (std::isinf(v) && v > 0.)
    return v;
  if (v == -0.f)
    v = 0.f;
  uint32_t ui = floatToBits(v);
  if (v >= 0)
    --ui;
  else
    ++ui;
  return bitsToFloat(ui);
}

/// Computes the next greater representable floating-point value
/// \param v floating point value
/// \return the next greater floating point value
inline double nextDoubleUp(double v) {
  if (std::isinf(v) && v > 0.)
    return v;
  if (v == -0.f)
    v = 0.f;
  uint64_t ui = doubleToBits(v);
  if (v >= 0)
    ++ui;
  else
    --ui;
  return bitsToDouble(ui);
}

/// Computes the next smaller representable floating-point value
/// \param v floating point value
/// \return the next smaller floating point value
inline double nextDoubleDown(double v) {
  if (std::isinf(v) && v > 0.)
    return v;
  if (v == -0.f)
    v = 0.f;
  uint64_t ui = doubleToBits(v);
  if (v >= 0)
    --ui;
  else
    ++ui;
  return bitsToDouble(ui);
}

} // namespace helios

#endif HELIOS_COMMON_UTILS_H
