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
///\file staggered_grid.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-18
///
///\brief

#include <poseidon/numeric/staggered_grid.h>

using namespace ponos;

namespace poseidon {

StaggeredGrid2Accessor::StaggeredGrid2Accessor(const ponos::size2 &resolution,
                                               const ponos::vec2 &spacing,
                                               ponos::Grid2Accessor<real_t> u,
                                               ponos::Grid2Accessor<real_t> v)
    : VectorGrid2Accessor(resolution, spacing, u, v) {}

ponos::vec2 StaggeredGrid2Accessor::operator()(const ponos::index2 &ij) {
  return 0.5f * vec2(this->u_(ij.plus(1, 0)) + this->u_(ij),
                     this->v_(ij.plus(0, 1)) + this->v_(ij));
}

ponos::vec2 StaggeredGrid2Accessor::operator()(const ponos::point2 &wp) {
  return ponos::vec2(this->u_(wp), this->v_(wp));
}

StaggeredGrid2::StaggeredGrid2() {
  StaggeredGrid2::setSpacing(vec2f(1.f));
  StaggeredGrid2::setOrigin(point2f(0.f));
}

StaggeredGrid2::StaggeredGrid2(ponos::size2 res,
                               ponos::vec2 s,
                               ponos::point2 o) {
  this->origin_ = o;
  this->resolution_ = res;
  this->spacing_ = s;
  this->u_grid.setResolution(res + size2(1, 0));
  this->u_grid.setOrigin(o + s.x * vec2(-0.5f, 0.f));
  this->u_grid.setSpacing(s);
  this->v_grid.setResolution(res + size2(0, 1));
  this->v_grid.setOrigin(o + s.y * vec2(0.f, -0.5f));
  this->v_grid.setSpacing(s);
}

void StaggeredGrid2::setSpacing(const vec2 &s) {
  this->spacing_ = s;
  this->u_grid.setSpacing(s);
  this->v_grid.setSpacing(s);
  setOrigin(this->origin_);
}

void StaggeredGrid2::resize(const size2 &res) {
  this->resolution_ = res;
  this->u_grid.setResolution(res + size2(1, 0));
  this->v_grid.setResolution(res + size2(0, 1));
}

void StaggeredGrid2::setOrigin(const point2f &o) {
  this->origin_ = o;
  this->u_grid.setOrigin(o + this->spacing_.x * vec2(-0.5f, 0.f));
  this->v_grid.setOrigin(o + this->spacing_.y * vec2(0.f, -0.5f));
}

StaggeredGrid2Accessor StaggeredGrid2::accessor(ponos::AddressMode address_mode,
                                                ponos::InterpolationMode interpolation_mode,
                                                real_t border) {
  return {this->resolution_, this->spacing_,
          this->u_grid.accessor(address_mode, interpolation_mode, border),
          this->v_grid.accessor(address_mode, interpolation_mode, border)};
}
} // poseidon namespace