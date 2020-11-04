/// Copyright (c) 2020, FilipeCN.
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
/// \file grid_interpolator.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-11-03
///
/// \brief

#ifndef PONOS_PONOS_PONOS_NUMERIC_GRID_INTERPOLATOR_H
#define PONOS_PONOS_PONOS_NUMERIC_GRID_INTERPOLATOR_H

#include <ponos/numeric/vector_grid.h>

namespace ponos {

/// Interpolates the field stored in a numerical grid
template<typename T>
class Grid1InterpolatorInterface {
public:
  Grid1InterpolatorInterface() = default;
  virtual ~Grid1InterpolatorInterface() = default;
  /// \param field
  virtual void setGrid(const ConstGrid1Accessor <T> &field) { field_ = field; }
  virtual T operator()(const ponos::point2 &wp) = 0;
protected:
  ConstGrid1Accessor <T> field_;
};

///
/// \tparam T
template<typename T>
class LinearGrid1Interpolator : public Grid1InterpolatorInterface<T> {
public:
  LinearGrid1Interpolator() = default;
  explicit LinearGrid1Interpolator(ConstGrid1Accessor <T> &&field) { this->field_ = field; }
  ~LinearGrid1Interpolator() override = default;
  /// \param world_position (in world coordinates)
  /// \return sampled data value in ``world_position`` based on the
  T operator()(const ponos::point2 &world_position) override {
    auto cp = this->field_.cellPosition(world_position);
    auto ip = this->cellIndex(world_position);
    return bilerp<T>(cp.x, cp.y, (*this)[ip], (*this)[ip.right()],
                     (*this)[ip.plus(1, 1)], (*this)[ip.up()]);
  }
};

template<typename T>
class MonotonicCubicGrid1Interpolator : public Grid1InterpolatorInterface<T> {
public:
  MonotonicCubicGrid1Interpolator() = default;
  explicit MonotonicCubicGrid1Interpolator(ConstGrid1Accessor <T> &&field) { this->field_ = field; }
  ~MonotonicCubicGrid1Interpolator() override = default;
  /// \param world_position (in world coordinates)
  /// \return sampled data value in ``world_position`` based on the
  T operator()(const ponos::point2 &world_position) override {
    auto cp = this->field_.cellPosition(world_position);
    auto ip = this->cellIndex(world_position);
    T f[4][4];
    for (int dx = -1; dx <= 2; ++dx)
      for (int dy = -1; dy <= 2; ++dy)
        f[dx + 1][dy + 1] = (*this)[ip + index2(dx, dy)];
    return monotonicCubicInterpolate(f, cp);
  }
};

template<typename T>
class LagrangeGrid1Interpolator : public Grid1InterpolatorInterface<T> {
public:
  LagrangeGrid1Interpolator() = default;
  ~LagrangeGrid1Interpolator() = default;
  void setK(u32 k) { k_ = k; }
  T operator()(const ponos::point2 &world_position) override {
    auto cp = this->field_.cellPosition(world_position);
    auto ip = this->cellIndex(world_position);
    i32 start = std::max(0, std::min(ip - k_ / 2, this->field_.resolution() - k_ - 1));
  }
private:
  i32 k_{4};
};

class Grid2Interpolator {

};

}

#endif //PONOS_PONOS_PONOS_NUMERIC_GRID_INTERPOLATOR_H
