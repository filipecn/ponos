// Created by filipecn on 2018-12-11.
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

#ifndef HELIOS_COMMON_E_FLOAT_H
#define HELIOS_COMMON_E_FLOAT_H

#include <limits>
#include <ponos/common/defs.h>
#include <ponos/ponos.h>

namespace helios {

static constexpr real_t machineEpsilon =
    std::numeric_limits<real_t>::epsilon() * 0.5;

inline constexpr real_t gammaBound(int n) {
  return (n * machineEpsilon) / (1 - n * machineEpsilon);
}

/// Implements the running error analysis by carrying the error bounds
/// accumulated by a floating point value. It keeps track of the interval of
/// uncertainty of the computed value.
class EFloat {
public:
  EFloat();
  /// \param v floating point value
  /// \param err absolute error bound
  explicit EFloat(float v, float err = 0.f);
  /// \return a bound for the absolute error
  float absoluteError() const;
  /// \return lower error interval bound
  float upperBound() const;
  /// \return upper error interval bound
  float lowerBound() const;
#ifdef NDEBUG
  /// \return relative error
  double relativeError() const;
  /// \return highly precise value
  loong double preciseValue() const;
#endif
  explicit operator float() const;
  EFloat operator+(EFloat f) const;
  EFloat operator-(EFloat f) const;
  EFloat operator*(EFloat f) const;
  EFloat operator/(EFloat f) const;
  bool operator==(EFloat f) const;
private:
  friend bool solve_quadratic(EFloat A, EFloat B, EFloat C, EFloat *t0,
                              EFloat *t1);
  float v;                    //!< computed value
  ponos::Interval<float> err; //!< absolute error bound
#ifdef NDEBUG
  long double ld; //!< highly precise version of v
#endif
};

inline EFloat operator*(float f, EFloat fe) { return EFloat(f) * fe; }

inline EFloat operator/(float f, EFloat fe) { return EFloat(f) / fe; }

inline EFloat operator+(float f, EFloat fe) { return EFloat(f) + fe; }

inline EFloat operator-(float f, EFloat fe) { return EFloat(f) - fe; }

inline bool solve_quadratic(EFloat A, EFloat B, EFloat C, EFloat *t0,
                            EFloat *t1) {
  // Find quadratic discriminant
  double discrim = (double)B.v * (double)B.v - 4. * (double)A.v * (double)C.v;
  if (discrim < 0.)
    return false;
  double rootDiscrim = std::sqrt(discrim);

  EFloat floatRootDiscrim(rootDiscrim, machineEpsilon * rootDiscrim);

  // Compute quadratic _t_ values
  EFloat q;
  if ((float)B < 0)
    q = -.5 * (B - floatRootDiscrim);
  else
    q = -.5 * (B + floatRootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if ((float)*t0 > (float)*t1)
    std::swap(*t0, *t1);
  return true;
}

} // namespace helios

#endif // HELIOS_COMMON_E_FLOAT_H