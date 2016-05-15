#pragma once

#include <cmath>

namespace ponos {

  #define PI 3.14159265359

  #define TO_RADIANS(A) \
    ((A) * 180.f / PI)

  template <typename T>
  T clamp(const T& n, const T& l, const T& u) {
    return std::max(l, std::min(n, u));
  }

}; // ponos namespace
