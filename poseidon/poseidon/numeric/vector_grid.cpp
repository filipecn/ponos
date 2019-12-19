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
///\file vector_grid.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-18
///
///\brief

#include <poseidon/numeric/vector_grid.h>

namespace poseidon {

VectorGrid2Accessor::VectorGrid2Accessor(const ponos::size2 &resolution,
                                         const ponos::vec2 &spacing,
                                         ponos::Grid2Accessor<real_t> u,
                                         ponos::Grid2Accessor<real_t> v)
    : resolution(resolution), spacing(spacing), u_(u), v_(v) {}

ponos::vec2 VectorGrid2Accessor::operator()(const ponos::index2 &ij) {
  return ponos::vec2(u_(ij), v_(ij));
}

ponos::vec2 VectorGrid2Accessor::operator()(const ponos::point2 &wp) {
  return ponos::vec2(u_(wp), v_(wp));
}

real_t &VectorGrid2Accessor::u(const ponos::index2 &ij) { return u_(ij); }

real_t &VectorGrid2Accessor::v(const ponos::index2 &ij) { return v_(ij); }

ponos::Grid2Accessor<float> &VectorGrid2Accessor::uAccessor() { return u_; }

ponos::Grid2Accessor<float> &VectorGrid2Accessor::vAccessor() { return v_; }

VectorGrid2::VectorGrid2() {
  VectorGrid2::setSpacing(ponos::vec2(1.f));
  VectorGrid2::setOrigin(ponos::point2(0.f));
}

VectorGrid2::VectorGrid2(const ponos::size2 &res,
                         const ponos::vec2 &s,
                         const ponos::point2 &o)
    : resolution_(res), origin_(o), spacing_(s) {
  u_grid.setResolution(res);
  u_grid.setOrigin(o);
  u_grid.setSpacing(s);
  v_grid.setResolution(res);
  v_grid.setOrigin(o);
  v_grid.setSpacing(s);
}

ponos::size2 VectorGrid2::resolution() const { return resolution_; }

ponos::vec2 VectorGrid2::spacing() const { return spacing_; }

ponos::point2 VectorGrid2::origin() const { return origin_; }

void VectorGrid2::resize(const ponos::size2 &res) {
  resolution_ = res;
  u_grid.setResolution(res);
  v_grid.setResolution(res);
}

void VectorGrid2::setOrigin(const ponos::point2 &o) {
  origin_ = o;
  u_grid.setOrigin(o);
  v_grid.setOrigin(o);
}

void VectorGrid2::setSpacing(const ponos::vec2 &s) {
  spacing_ = s;
  u_grid.setSpacing(s);
  v_grid.setSpacing(s);
}

const ponos::Grid2<real_t> &VectorGrid2::u() const { return u_grid; }

const ponos::Grid2<real_t> &VectorGrid2::v() const { return v_grid; }

VectorGrid2Accessor VectorGrid2::accessor(ponos::AddressMode address_mode,
                                          ponos::InterpolationMode interpolation_mode,
                                          real_t border) {
  return {resolution_, spacing_, u_grid.accessor(
      address_mode, interpolation_mode, border),
          v_grid.accessor(address_mode, interpolation_mode, border)};
}

} // poseidon namespace