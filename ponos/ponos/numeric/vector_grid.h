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
/// \file vector_grid.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-02-17
///
/// \brief

#ifndef PONOS_NUMERIC_VECTOR_GRID_H
#define PONOS_NUMERIC_VECTOR_GRID_H

#include <ponos/numeric/grid.h>

namespace ponos {

/// \brief Vector components distribution in the grid
///
/// CELL_CENTERED: Vector components are stored at cell centers
///                 ---------
///                |         |
///                |  (u,v)  |
///                |         |
///                ----------
/// STAGGERED:
///                 ----------
///                |         |
///                u         |
///                |  w      |
///                -----v----
enum class VectorGridType { CELL_CENTERED, STAGGERED };

template <typename T> class VectorGrid2;

template <typename T> class VectorGrid2Accessor {
public:
  /// \param grid_type **[in]**
  /// \param u_acc **[in]**
  /// \param v_acc **[in]**
  VectorGrid2Accessor(VectorGrid2<T> &grid, Grid2Accessor<T> u_acc,
                      Grid2Accessor<T> v_acc)
      : grid_(grid), u_(u_acc), v_(v_acc) {}
  Grid2Accessor<T> &u() { return u_; }
  Grid2Accessor<T> &v() { return v_; }
  /// \return grid resolution
  size2 resolution() const { return grid_.resolution_; }
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
  /// \param ij **[in]**
  /// \return Vector2<T>
  Vector2<T> operator[](const index2 &ij) {
    switch (grid_.grid_type_) {
    case VectorGridType::CELL_CENTERED:
      return Vector2<T>(u_[ij], v_[ij]);
    case VectorGridType::STAGGERED:
      return Vector2<T>((u_[ij] + u_[ij.right()]) / 2,
                        (v_[ij] + v_[ij.up()]) / 2);
    }
    return Vector2<T>();
  }
  /// \param wp **[in]**
  /// \return Vector2<T>
  Vector2<T> operator()(const point2 &wp) { return {u_(wp), v_(wp)}; }

private:
  VectorGrid2<T> &grid_;
  Grid2Accessor<T> u_, v_;
};

/// Represents a vector field. Data samples follow the vector grid type.
/// The origin is at cell centes
///                 ---------
///                |         |
///                |    o    |
///                |         |
///                ----------
/// For a NxM vector grid, the staggered type stores (N + 1)xM v components and
/// Nx(M+1) u components
/// \tparam T
template <typename T> class VectorGrid2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "VectorGrid2 must hold an float type!");

public:
  friend class VectorGrid2Accessor<T>;
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  ///
  VectorGrid2(VectorGridType grid_type = VectorGridType::CELL_CENTERED)
      : grid_type_(grid_type) {
    setSpacing(vec2(1.f));
    setOrigin(point2(0.f));
  }
  /// \param res **[in]** resolution (in number of cells)
  /// \param o **[in]** origin (0,0) corner cell center
  /// \param s **[in]** cell size
  /// \param grid_type **[in]** vector components position
  VectorGrid2(size2 res, point2 o, vec2 s,
              VectorGridType grid_type = VectorGridType::CELL_CENTERED)
      : grid_type_(grid_type), resolution_(res), origin_(o), spacing_(s) {
    setResolution(res);
    setSpacing(s);
    setOrigin(o);
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// Copy all fields from other object. Performs raw data copy.
  /// \param other
  VectorGrid2 &operator=(const VectorGrid2 &other) {
    grid_type_ = other.grid_type_;
    setResolution(other.resolution_);
    setSpacing(other.spacing_);
    setOrigin(other.origin_);
    u_ = other.u_;
    v_ = other.v_;
    return *this;
  }
  /// Assign value to all positions
  /// \param value assign value
  /// \return *this
  VectorGrid2 &operator=(const T &value) {
    u_ = value;
    v_ = value;
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \return size2 grid resolution
  size2 resolution() const { return resolution_; }
  /// \return vec2f grid spacing (cell size)
  vec2 spacing() const { return spacing_; }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return origin_; }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void setResolution(size2 res) {
    resolution_ = res;
    switch (grid_type_) {
    case VectorGridType::CELL_CENTERED:
      u_.setResolution(res);
      v_.setResolution(res);
      break;
    case VectorGridType::STAGGERED:
      u_.setResolution(res + size2(1, 0));
      v_.setResolution(res + size2(0, 1));
      break;
    default:
      break;
    }
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point2 &o) {
    origin_ = o;
    switch (grid_type_) {
    case VectorGridType::CELL_CENTERED:
      u_.setOrigin(o);
      v_.setOrigin(o);
      break;
    case VectorGridType::STAGGERED:
      u_.setOrigin(o - vec2(spacing_.x / 2, 0));
      v_.setOrigin(o - vec2(0, spacing_.y / 2));
      break;

    default:
      break;
    }
    updateTransform();
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setSpacing(const vec2 &s) {
    spacing_ = s;
    u_.setSpacing(s);
    v_.setSpacing(s);
    setOrigin(origin_);
  }
  const Grid2<float> &u() const { return u_; }
  const Grid2<float> &v() const { return v_; }
  Grid2<float> &u() { return u_; }
  Grid2<float> &v() { return v_; }
  /// \param address_mode **[in]**
  /// \param interpolation_mode **[in]**
  /// \param border **[in]**
  /// \return VectorGrid2Accessor<T>
  VectorGrid2Accessor<T> accessor(
      AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
      InterpolationMode interpolation_mode = InterpolationMode::MONOTONIC_CUBIC,
      T border = T(0)) {
    return VectorGrid2Accessor<T>(
        *this, u_.accessor(address_mode, interpolation_mode, border),
        v_.accessor(address_mode, interpolation_mode, border));
  }

protected:
  void updateTransform() {
    to_world_ = translate(vec2(origin_.x, origin_.y)) * scale(spacing_);
    to_grid_ = inverse(to_world_);
  }
  Transform2 to_grid_, to_world_;
  VectorGridType grid_type_{VectorGridType::CELL_CENTERED};
  size2 resolution_;
  point2 origin_;
  vec2 spacing_;
  Grid2<T> u_, v_;
};

} // namespace ponos

#endif