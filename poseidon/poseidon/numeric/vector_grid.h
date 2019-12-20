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
///\file vector_grid.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-18
///
///\brief

#ifndef POSEIDON_NUMERIC_VECTOR_GRID_H
#define POSEIDON_NUMERIC_VECTOR_GRID_H

#include <ponos/numeric/grid.h>

namespace poseidon {

/// Class that provides access to vector grid elements
class VectorGrid2Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  VectorGrid2Accessor(const ponos::size2 &resolution,
                      const ponos::vec2 &spacing,
                      ponos::Grid2Accessor<real_t> u,
                      ponos::Grid2Accessor<real_t> v);
  /// \param ij
  /// \return
  virtual ponos::vec2 operator()(const ponos::index2 &ij);
  /// \param wp
  /// \return
  virtual ponos::vec2 operator()(const ponos::point2 &wp);
  /// \param ij
  /// \return
  real_t &u(const ponos::index2 &ij);
  /// \param ij
  /// \return
  real_t &v(const ponos::index2 &ij);
  /// \return
  ponos::Grid2Accessor<float> &uAccessor();
  /// \return
  ponos::Grid2Accessor<float> &vAccessor();

  const ponos::size2 resolution;
  const ponos::vec2 spacing;

protected:
  ponos::Grid2Accessor<real_t> u_, v_;
};
/// The vector grid stores a vector in each grid point. Useful to represent
/// velocity fields.
class VectorGrid2 {
public:
  VectorGrid2();
  /// \param res resolution in number of cells
  /// \param s spacing cell size
  /// \param o origin (0,0,0) corner position
  VectorGrid2(const ponos::size2 &res,
              const ponos::vec2 &s,
              const ponos::point2 &o = ponos::point2());
  void copy(const VectorGrid2 &other);
  ponos::size2 resolution() const;
  ponos::vec2 spacing() const;
  ponos::point2 origin() const;
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(const ponos::size2 &res);
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const ponos::point2 &o);
  /// Changes grid cell size
  /// \param d new size
  virtual void setSpacing(const ponos::vec2 &s);
  /// \return u component accessor
  ponos::Grid2Accessor<real_t> u();
  /// \return v component accessor
  ponos::Grid2Accessor<real_t> v();
  ///
  /// \param address_mode
  /// \param interpolation_mode
  /// \param border
  /// \return
  VectorGrid2Accessor accessor(ponos::AddressMode address_mode = ponos::AddressMode::CLAMP_TO_EDGE,
                               ponos::InterpolationMode interpolation_mode = ponos::InterpolationMode::MONOTONIC_CUBIC,
                               real_t border = 0);
  /// Assigns value to every u and v component
  /// \param value assign value
  /// \return *this
  VectorGrid2 &operator=(const real_t &value);
  /// Assigns value.x to every u component and value.y to every v component
  /// \param value assign value
  /// \return *this
  VectorGrid2 &operator=(const ponos::vec2 &value);

protected:
  ponos::size2 resolution_;
  ponos::point2 origin_;
  ponos::vec2 spacing_;
  ponos::Grid2<real_t> u_grid_, v_grid_;
};

} // poseidon namespace

#endif //POSEIDON_NUMERIC_VECTOR_GRID_H