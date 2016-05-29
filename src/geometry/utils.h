#pragma once

#include <cmath>

namespace ponos {

  #define PI 3.14159265359
  #define PI_2 6.28318530718

  #define TO_RADIANS(A) \
    ((A) * 180.f / PI)

  #define IS_ZERO(A) \
    (fabs(A) < 1e-8)

  #define IS_EQUAL(A, B) \
    (fabs((A) - (B)) < 1e-8)

  template <typename T>
  T clamp(const T& n, const T& l, const T& u) {
    return std::max(l, std::min(n, u));
  }

  template <typename T>
  void swap(T &a, T &b) {
    T tmp = b;
    b = a;
    a = tmp;
  }

  inline bool solve_quadratic(float A, float B, float C, float &t0, float &t1) {
    float delta = B * B - 4.f * A * C;
    if(IS_ZERO(A) || delta <= 0.)
      return false;
    float sDelta = sqrtf(delta);
    float q;
    if(B < 0.)
      q = -0.5 * (B - sDelta);
    else q = -0.5f * (B + sDelta);
    t0 = q / A;
    t1 = C / q;
    if(t0 > t1)
      swap(t0, t1);
    return true;
  }

}; // ponos namespace
