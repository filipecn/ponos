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
///\file grid.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-13
///
///\brief

#ifndef PONOS_NUMERIC_GRID_H
#define PONOS_NUMERIC_GRID_H

#include <ponos/geometry/transform.h>
#include <ponos/numeric/interpolation.h>
#include <ponos/storage/array.h>

namespace ponos {

/// \brief Position names inside the grid
///  CELL_CENTER: cell centers (c)
///  U_FACE_CENTER: vertical face centers (u)
///  V_FACE_CENTER: horizontal face centers (v)
///  W_FACE_CENTER: depth face centers (w)
///  VERTEX_CENTER: vertex centers (V)
///
///  Diagram of positions:
///
///           ------------
///           |          |
///           u    c     |
///           |          |
///           V----v------
///
enum class GridPosition {
  CELL_CENTER,
  U_FACE_CENTER,
  V_FACE_CENTER,
  W_FACE_CENTER,
  VERTEX_CENTER
};
/// Grid address mode: Defines how out of boundaries grid positions are handled
///
/// REPEAT:
/// CLAMP_TO_EDGE:
/// BORDER:
/// WRAP:
/// MIRROR:
enum class AddressMode { REPEAT, CLAMP_TO_EDGE, BORDER, WRAP, MIRROR, NONE };
enum class FilterMode { LINEAR, POINT };
/// The type of interpolation used when sampling arbitrary positions over a grid
enum class InterpolationMode { LINEAR, MONOTONIC_CUBIC };

// forward declaration of Grid2
template <typename T> class Grid2;
// forward declaration of Grid2Accessor
template <typename T> class Grid2Accessor;
/// Auxiliary class to allow c++ iteration in loops.
/// Ex: for(auto e : grid.accessor()) {}
/// \tparam T grid data type
template <typename T> class Grid2Iterator {
public:
  /// Represents the current grid index element being iterated
  class Element {
  public:
    Element(T &v, const index2 &ij, const Grid2Accessor<T> &acc)
        : value(v), index(ij), acc_(acc) {}
    /// \return world position coordinates of the current index
    point2 worldPosition() const { return acc_.worldPosition(index); }
    /// \return region (in world coordinates) of the grid cell respective to
    /// the current index
    bbox2 region() const { return acc_.cellRegion(index); }
    /// Reference to the grid data stored in the current index
    T &value;
    /// Index of the grid data element
    const index2 &index;

  private:
    const Grid2Accessor<T> &acc_;
  };
  ///
  /// \param grid_accessor grid to be iterated over
  /// \param ij starting grid index for iteration
  Grid2Iterator(Grid2Accessor<T> &grid_accessor, const index2 &ij)
      : acc_(grid_accessor),
        it(Index2Iterator<i32>(index2(0, 0),
                               index2(grid_accessor.resolution().width,
                                      grid_accessor.resolution().height),
                               ij)) {}
  /// \return increment operator to move to the next grid index
  Grid2Iterator &operator++() {
    ++it;
    return *this;
  }
  /// \return current grid index element
  Element operator*() { return Element(acc_[*it], *it, acc_); }
  /// operator required by the loop iteration code of c++
  /// \param other
  /// \return true if iterators are different
  bool operator!=(const Grid2Iterator<T> &other) {
    return acc_.resolution() != other.acc_.resolution() || *it != *other.it;
  }

private:
  Grid2Accessor<T> &acc_;
  Index2Iterator<i32> it;
};
/// Class that provides access to grid elements.
/// \tparam T grid data type
template <typename T> class Grid2Accessor {
public:
  /// \param grid
  /// \param address_mode
  /// \param border
  explicit Grid2Accessor(
      Grid2<T> &grid, AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
      T border = T(0),
      InterpolationMode interpolation_mode = InterpolationMode::MONOTONIC_CUBIC)
      : grid_(grid), address_mode_(address_mode),
        interpolation_mode_(interpolation_mode), border_(border) {}
  /// \return grid resolution
  size2 resolution() const { return grid_.data_.size(); }
  /// \return grid spacing
  vec2 spacing() const { return grid_.spacing_; }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return grid_.origin_; }
  /// \param world_position (in world coordinates)
  /// \return world_position in grid coordinates
  point2 gridPosition(const point2 &world_position) const {
    return grid_.to_grid_(world_position);
  }
  /// \param grid_position (in grid coordinates)
  /// \return grid_position in world coordinates
  point2 worldPosition(const point2 &grid_position) const {
    return grid_.to_world_(grid_position);
  }
  /// \param grid_position (in grid coordinates)
  /// \return grid_position in world coordinates
  point2 worldPosition(const index2 &grid_position) const {
    return grid_.to_world_(point2(grid_position.i, grid_position.j));
  }
  /// \param world_position (in world coordinates)
  /// \return offset of world_position inside the cell containing it (in [0,1]).
  point2 cellPosition(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return gp - vec2(static_cast<int>(gp.x), static_cast<int>(gp.y));
  }
  /// \param world_position (in world coordinates)
  /// \return index of the cell that contains world_position
  index2 cellIndex(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return index2(gp.x, gp.y);
  }
  /// \param ij cell index
  /// \return ij cell's region (in world coordinates)
  bbox2 cellRegion(const index2 &ij) const {
    auto wp = grid_.to_world_(point2(ij.i, ij.j));
    return bbox2(wp, wp + grid_.spacing_);
  }
  /// \param ij position index
  /// \return reference to data stored at ij based on the address mode
  T &operator[](const index2 &ij) {
    index2 fij = ij;
    switch (address_mode_) {
    case AddressMode::REPEAT:
      fij.i = (ij.i < 0) ? grid_.resolution().width - 1 - ij.i
                         : ij.i % grid_.resolution().width;
      fij.j = (ij.j < 0) ? grid_.resolution().height - 1 - ij.j
                         : ij.j % grid_.resolution().height;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      fij.clampTo(
          size2(grid_.resolution().width - 1, grid_.resolution().height - 1));
      break;
    case AddressMode::BORDER:
      if (!grid_.data().stores(ij)) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case AddressMode::WRAP:
      break;
    case AddressMode::MIRROR:
      break;
    default:
      break;
    }
    FATAL_ASSERT(grid_.data().stores(fij));
    return grid_.data()[fij];
  }
  /// \param world_position (in world coordinates)
  /// \return sampled data value in world_position based on the interpolation
  /// mode and address mode
  T operator()(const point2 &world_position) {
    auto cp = cellPosition(world_position);
    auto ip = cellIndex(world_position);
    switch (interpolation_mode_) {
    case InterpolationMode::LINEAR:
      return bilerp<T>(cp.x, cp.y, (*this)[ip], (*this)[ip + index2(1, 0)],
                       (*this)[ip + index2(1, 1)], (*this)[ip + index2(0, 1)]);
    case InterpolationMode::MONOTONIC_CUBIC:
      T f[4][4];
      for (int dx = -1; dx <= 2; ++dx)
        for (int dy = -1; dy <= 2; ++dy)
          f[dx + 1][dy + 1] = (*this)[ip + index2(dx, dy)];
      return monotonicCubicInterpolate(f, cp);
    default:
      break;
    }
    return T(0);
  }
  /// \return starting iterator
  Grid2Iterator<T> begin() { return Grid2Iterator<float>(*this, index2(0, 0)); }
  /// \return sentinel iterator
  Grid2Iterator<T> end() {
    return Grid2Iterator<float>(
        *this, index2(grid_.resolution().width, grid_.resolution().height));
  }

private:
  Grid2<T> &grid_;
  AddressMode address_mode_ = AddressMode::CLAMP_TO_EDGE;
  InterpolationMode interpolation_mode_ = InterpolationMode::LINEAR;
  T border_ = T(0);
  T dummy_ = T(0);
};
/// A Grid2 is a numerical discretization of an arbitrary 2-dimensional domain
/// that can be used for numerical simulations and other applications.
/// The grid origin (O) is based on the lowest cell center of the grid
///                           .        .
///                           .      .
///                          --- ---
///                         | O |   | ...
///                          --------
/// Cell indices are cell-based, so the lowest vertex of cell (i,j), in index
/// space, is (i - 0.5, j - 0.5).
/// \tparam T grid data type
template <typename T> class Grid2 {
public:
  friend class Grid2Accessor<T>;
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Grid2() = default;
  /// \param resolution grid resolution
  /// \param spacing cell size
  /// \return grid origin (world position of index (0,0))
  explicit Grid2(const size2 &resolution, const vec2 &spacing = vec2(1, 1),
                 const point2 &origin = point2(0, 0))
      : origin_(origin), spacing_(spacing) {
    setResolution(resolution);
    updateTransform();
  }
  Grid2(const Grid2 &other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = other.data_;
  }
  Grid2(Grid2 &&other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = std::move(other.data_);
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// Copy all fields from other object. Performs raw data copy.
  /// \param other
  Grid2 &operator=(const Grid2 &other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = other.data_;
    return *this;
  }
  /// Assign value to all positions
  /// \param value assign value
  /// \return *this
  Grid2 &operator=(const T &value) {
    data_ = value;
    return *this;
  }
  Grid2 &operator=(Array2<T> &values) {
    data_ = values;
    return *this;
  }
  Grid2 &operator=(Array2<T> &&values) {
    data_ = values;
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \return grid resolution (in cells)
  size2 resolution() const { return data_.size(); }
  /// \return grid spacing (cell size)
  vec2 spacing() const { return spacing_; }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return origin_; }
  /// \param new_resolution
  void setResolution(const size2 &new_resolution) {
    data_.resize(new_resolution);
  }
  /// \param new_spacing
  void setSpacing(const vec2 &new_spacing) {
    spacing_ = new_spacing;
    updateTransform();
  }
  /// \param new_origin
  void setOrigin(const point2 &new_origin) {
    origin_ = new_origin;
    updateTransform();
  }
  /// \return reference to memory data
  const Array2<T> &data() const { return data_; }
  /// \return reference to memory data
  Array2<T> &data() { return data_; }
  /// \param address_mode
  /// \param border border value
  /// \return accessor for grid data
  Grid2Accessor<T> accessor(
      AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
      InterpolationMode interpolation_mode = InterpolationMode::MONOTONIC_CUBIC,
      T border = T(0)) {
    return Grid2Accessor<T>(*this, address_mode, border, interpolation_mode);
  }

private:
  void updateTransform() {
    to_world_ = translate(vec2(origin_.x, origin_.y)) * scale(spacing_);
    to_grid_ = inverse(to_world_);
  }
  Transform2 to_grid_, to_world_;
  point2 origin_;
  vec2 spacing_;
  Array2<T> data_;
};

} // namespace ponos

#endif // PONOS_NUMERIC_GRID_H