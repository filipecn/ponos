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
///\file staggered_grid.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-18
///
///\brief

#ifndef POSEIDON_NUMERIC_STAGGERED_GRID_H
#define POSEIDON_NUMERIC_STAGGERED_GRID_H

#include <poseidon/numeric/vector_grid.h>

namespace poseidon {

class StaggeredGrid2Accessor : public VectorGrid2Accessor {
public:
  StaggeredGrid2Accessor(const ponos::size2 &resolution,
                         const ponos::vec2 &spacing,
                         ponos::Grid2Accessor<real_t> u,
                         ponos::Grid2Accessor<real_t> v);
  ponos::vec2 operator()(const ponos::index2 &ij) override;
  ponos::vec2 operator()(const ponos::point2 &wp) override;
};

/// Represents a staggered grid with regular grid
/// The origin of each staggered grid follows the scheme:
///    ----------
///   |         |
///   u    o    |
///   |  w      |
///   -----v----
class StaggeredGrid2 : public VectorGrid2 {
public:
  StaggeredGrid2();
  /// \param res resolution in number of cells
  /// \param o origin (0,0) corner position
  /// \param s cell size
  StaggeredGrid2(ponos::size2 res,
                 ponos::vec2 s,
                 ponos::point2 o = ponos::point2());
  /// Changes grid cell size
  /// \param d new size
  void setSpacing(const ponos::vec2 &s) override;
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(const ponos::size2 &res) override;
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const ponos::point2 &o) override;
  StaggeredGrid2Accessor accessor(ponos::AddressMode address_mode = ponos::AddressMode::CLAMP_TO_EDGE,
                                  ponos::InterpolationMode interpolation_mode = ponos::InterpolationMode::MONOTONIC_CUBIC,
                                  real_t border = 0);
};

} // poseidon namespace

#endif //POSEIDON_NUMERIC_STAGGERED_GRID_H