/// Copyright (c) 2019, FilipeCN.
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
///\file vector_fields.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-19
///
///\brief

#include <poseidon/numeric/vector_fields.h>

namespace poseidon {

ponos::vec2 zalesakDeformationField(ponos::point2 p) {
  real_t pi_2 = ponos::Constants::pi * 0.5f;
  return ponos::vec2(-pi_2 * (0.5f - p.y), -pi_2 * (p.x - 0.5f));
}

ponos::vec3 enrightDeformationField(ponos::point3 p) {
  real_t pix = ponos::Constants::pi * p.x;
  real_t piy = ponos::Constants::pi * p.y;
  real_t piz = ponos::Constants::pi * p.z;
  return ponos::vec3(
      2. * sinf(pix) * sinf(pix) * sinf(2 * piy) * sinf(2 * piz),
      -sinf(2 * pix) * sinf(piy) * sinf(piy) * sinf(2 * piz),
      -sinf(2 * pix) * sinf(2 * piy) * sinf(piz) * sinf(piz));
}

} // poseidon namespace