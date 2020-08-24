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
///  - Diagram of positions:
///
///           - ----------
///           |          |
///           u    c     |
///           |          |
///           V----v------
///
enum class GridPosition {
  ///  cell centers (c)
  CELL_CENTER,
  ///  vertical face centers (u)
  U_FACE_CENTER,
  ///  horizontal face centers (v)
  V_FACE_CENTER,
  ///  depth face centers (w)
  W_FACE_CENTER,
  ///  vertex centers (V)
  VERTEX_CENTER,
  GRID_POSITION_TYPES_COUNT
};
/// Grid address mode: Defines how out of boundaries grid positions are handled
enum class AddressMode {
  ///
  REPEAT,
  /// exterior indices are projected to the nearest index in the grid
  CLAMP_TO_EDGE,
  /// exterior indices are receive an arbitrary value
  BORDER,
  ///
  WRAP,
  /// exterior indices are reflected into the grid
  MIRROR,
  NONE
};
///
enum class FilterMode { LINEAR, POINT };
/// The type of interpolation used when sampling arbitrary positions over a grid
enum class InterpolationMode { LINEAR, MONOTONIC_CUBIC };

// forward declaration of Grid2
template<typename T> class Grid2;
// forward declaration of Grid2Accessor
template<typename T> class Grid2Accessor;
template<typename T> class ConstGrid2Accessor;
/// Auxiliary class to allow c++ iteration in loops.
/// Ex: for(auto e : grid.accessor()) {}
/// \tparam T grid data type
template<typename T> class Grid2Iterator {
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
  Index2Iterator <i32> it;
};
/// Auxiliary class to allow c++ iteration in loops.
/// Ex: for(auto e : grid.accessor()) {}
/// \tparam T grid data type
template<typename T> class ConstGrid2Iterator {
public:
  /// Represents the current grid index element being iterated
  class Element {
  public:
    Element(const T &v, const index2 &ij, const ConstGrid2Accessor<T> &acc)
        : value(v), index(ij), acc_(acc) {}
    /// \return world position coordinates of the current index
    point2 worldPosition() const { return acc_.worldPosition(index); }
    /// \return region (in world coordinates) of the grid cell respective to
    /// the current index
    bbox2 region() const { return acc_.cellRegion(index); }
    /// Reference to the grid data stored in the current index
    const T &value;
    /// Index of the grid data element
    const index2 &index;

  private:
    const ConstGrid2Accessor<T> &acc_;
  };
  ///
  /// \param grid_accessor grid to be iterated over
  /// \param ij starting grid index for iteration
  ConstGrid2Iterator(ConstGrid2Accessor<T> &grid_accessor, const index2 &ij)
      : acc_(grid_accessor),
        it(Index2Iterator<i32>(index2(0, 0),
                               index2(grid_accessor.resolution().width,
                                      grid_accessor.resolution().height),
                               ij)) {}
  /// \return increment operator to move to the next grid index
  ConstGrid2Iterator &operator++() {
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
  ConstGrid2Accessor<T> &acc_;
  Index2Iterator <i32> it;
};
/// The Grid2Accessor provides access to grid elements following specific rules
/// defined on its construction. With the accessor it is possible to:
/// - Access out of bounds positions;
/// - Interpolate data on arbitrary positions;
/// - Iterate through grid cells in a convenient way:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       for(auto c : my_grid2accessor) {
///         c.value = 0; // cell value access
///         c.index; // cell index
///         c.worldPosition(); // cell center position in world coordinates
///         c.region(); // cell region (in world coordinates)
///       }
/// \endverbatim
/// \tparam T grid data type
template<typename T> class Grid2Accessor {
public:
  /// Constructor
  /// \param grid **[in]** reference to a Grid2 object
  /// \param address_mode **[in | default = CLAMP_TO_EDGE]** how out of bounds
  /// positions are handled by the accessor
  /// \param border **[in | default = 0]** border value (when AddressMode::BORDE
  /// is chosen)
  /// \param interpolation_mode **[in | default = MONOTONIC_CUBIC]**
  /// interpolation method used in sampling
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
  /// \param world_position **[in]** (in world coordinates)
  /// \return ``world_position`` in grid coordinates
  point2 gridPosition(const point2 &world_position) const {
    return grid_.to_grid_(world_position);
  }
  /// \param grid_position **[in]** (in grid coordinates)
  /// \return ``grid_position`` in world coordinates
  point2 worldPosition(const point2 &grid_position) const {
    return grid_.to_world_(grid_position);
  }
  /// \param grid_position **[in]** (in grid coordinates)
  /// \return ``grid_position`` in world coordinates
  point2 worldPosition(const index2 &grid_position) const {
    return grid_.to_world_(point2(grid_position.i, grid_position.j));
  }
  /// \param world_position **[in]** (in world coordinates)
  /// \return offset of ``world_position`` inside the cell containing it (in
  /// [0,1]).
  point2 cellPosition(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return gp - vec2(static_cast<int>(gp.x), static_cast<int>(gp.y));
  }
  /// \param world_position **[in]** (in world coordinates)
  /// \return index of the cell that contains world_position
  index2 cellIndex(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return index2(gp.x, gp.y);
  }
  /// \param ij **[in]** cell index
  /// \return ij cell's region (in world coordinates)
  bbox2 cellRegion(const index2 &ij) const {
    auto wp = grid_.to_world_(point2(ij.i, ij.j));
    return bbox2(wp, wp + grid_.spacing_);
  }
  /// \param ij **[in]** position index
  /// \return reference to data stored at ``ij`` based on the address mode
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
    case AddressMode::WRAP:break;
    case AddressMode::MIRROR:break;
    default:break;
    }
    FATAL_ASSERT(grid_.data().stores(fij));
    return grid_.data()[fij];
  }
  /// \param world_position (in world coordinates)
  /// \return sampled data value in ``world_position`` based on the
  /// interpolation mode and address mode
  T operator()(const point2 &world_position) {
    auto cp = cellPosition(world_position);
    auto ip = cellIndex(world_position);
    switch (interpolation_mode_) {
    case InterpolationMode::LINEAR:
      return bilerp<T>(cp.x, cp.y, (*this)[ip], (*this)[ip.right()],
                       (*this)[ip.plus(1, 1)], (*this)[ip.up()]);
    case InterpolationMode::MONOTONIC_CUBIC:T f[4][4];
      for (int dx = -1; dx <= 2; ++dx)
        for (int dy = -1; dy <= 2; ++dy)
          f[dx + 1][dy + 1] = (*this)[ip + index2(dx, dy)];
      return monotonicCubicInterpolate(f, cp);
    default:break;
    }
    return T(0);
  }
  // \return starting iterator
  Grid2Iterator<T> begin() { return Grid2Iterator<float>(*this, index2(0, 0)); }
  // \return sentinel iterator
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
/// The ConstGrid2Accessor provides access to const grid elements following specific
/// rules defined on its construction. With the accessor it is possible to:
/// - Access out of bounds positions;
/// - Interpolate data on arbitrary positions;
/// - Iterate through grid cells in a convenient way:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       for(auto c : my_grid2accessor) {
///         c.value = 0; // cell value access
///         c.index; // cell index
///         c.worldPosition(); // cell center position in world coordinates
///         c.region(); // cell region (in world coordinates)
///       }
/// \endverbatim
/// \tparam T grid data type
template<typename T> class ConstGrid2Accessor {
public:
  /// Constructor
  /// \param grid **[in]** reference to a Grid2 object
  /// \param address_mode **[in | default = CLAMP_TO_EDGE]** how out of bounds
  /// positions are handled by the accessor
  /// \param border **[in | default = 0]** border value (when AddressMode::BORDE
  /// is chosen)
  /// \param interpolation_mode **[in | default = MONOTONIC_CUBIC]**
  /// interpolation method used in sampling
  explicit ConstGrid2Accessor(
      const Grid2<T> &grid, AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
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
  /// \param world_position **[in]** (in world coordinates)
  /// \return ``world_position`` in grid coordinates
  point2 gridPosition(const point2 &world_position) const {
    return grid_.to_grid_(world_position);
  }
  /// \param grid_position **[in]** (in grid coordinates)
  /// \return ``grid_position`` in world coordinates
  point2 worldPosition(const point2 &grid_position) const {
    return grid_.to_world_(grid_position);
  }
  /// \param grid_position **[in]** (in grid coordinates)
  /// \return ``grid_position`` in world coordinates
  point2 worldPosition(const index2 &grid_position) const {
    return grid_.to_world_(point2(grid_position.i, grid_position.j));
  }
  /// \param world_position **[in]** (in world coordinates)
  /// \return offset of ``world_position`` inside the cell containing it (in
  /// [0,1]).
  point2 cellPosition(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return gp - vec2(static_cast<int>(gp.x), static_cast<int>(gp.y));
  }
  /// \param world_position **[in]** (in world coordinates)
  /// \return index of the cell that contains world_position
  index2 cellIndex(const point2 &world_position) const {
    auto gp = grid_.to_grid_(world_position);
    return index2(gp.x, gp.y);
  }
  /// \param ij **[in]** cell index
  /// \return ij cell's region (in world coordinates)
  bbox2 cellRegion(const index2 &ij) const {
    auto wp = grid_.to_world_(point2(ij.i, ij.j));
    return bbox2(wp, wp + grid_.spacing_);
  }
  /// \param ij **[in]** position index
  /// \return reference to data stored at ``ij`` based on the address mode
  const T &operator[](const index2 &ij) {
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
    case AddressMode::WRAP:break;
    case AddressMode::MIRROR:break;
    default:break;
    }
    FATAL_ASSERT(grid_.data().stores(fij));
    return grid_.data()[fij];
  }
  /// \param world_position (in world coordinates)
  /// \return sampled data value in ``world_position`` based on the
  /// interpolation mode and address mode
  T operator()(const point2 &world_position) {
    auto cp = cellPosition(world_position);
    auto ip = cellIndex(world_position);
    switch (interpolation_mode_) {
    case InterpolationMode::LINEAR:
      return bilerp<T>(cp.x, cp.y, (*this)[ip], (*this)[ip.right()],
                       (*this)[ip.plus(1, 1)], (*this)[ip.up()]);
    case InterpolationMode::MONOTONIC_CUBIC:T f[4][4];
      for (int dx = -1; dx <= 2; ++dx)
        for (int dy = -1; dy <= 2; ++dy)
          f[dx + 1][dy + 1] = (*this)[ip + index2(dx, dy)];
      return monotonicCubicInterpolate(f, cp);
    default:break;
    }
    return T(0);
  }
  // \return starting iterator
  ConstGrid2Iterator<T> begin() { return ConstGrid2Iterator<float>(*this, index2(0, 0)); }
  // \return sentinel iterator
  ConstGrid2Iterator<T> end() {
    return ConstGrid2Iterator<float>(
        *this, index2(grid_.resolution().width, grid_.resolution().height));
  }

private:
  const Grid2<T> &grid_;
  AddressMode address_mode_ = AddressMode::CLAMP_TO_EDGE;
  InterpolationMode interpolation_mode_ = InterpolationMode::LINEAR;
  T border_ = T(0);
  T dummy_ = T(0);
};
/// A Grid2 is a numerical discretization of an arbitrary 2-dimensional domain
/// that can be used for numerical simulations and other applications.
/// - The grid origin (O) is based on the lowest cell center of the grid:
///
///                           .        .
///                           .      .
///                          --- ---
///                         | O |   | ...
///                          -------
/// - Cell indices are cell-center based, so the lowest vertex of cell
/// ``(i,j)``, in index space, is ``(i - 0.5, j - 0.5)``.
/// \verbatim embed:rst:leading-slashes"
///   .. note::
///     An Array2 object is used to store the data internally. Thus the index
///     system of the Grid2 is directly aligned to the convention used by
///     Array2. ``i`` for **x** and ``j`` for **y**.
/// \endverbatim
/// \tparam T grid data type
template<typename T> class Grid2 {
public:
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Grid must hold an float type!");
  friend class Grid2Accessor<T>;
  friend class ConstGrid2Accessor<T>;
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Grid2() = default;
  /// \param resolution **[in]** grid resolution (in cells count)
  /// \param spacing **[in | default = vec2(1,1)]** cell size
  /// \param origin **[in | default = point2(0,0)]** grid origin (world
  /// position of index (0,0))
  explicit Grid2(const size2 &resolution, const vec2 &spacing = vec2(1, 1),
                 const point2 &origin = point2(0, 0))
      : origin_(origin), spacing_(spacing) {
    setResolution(resolution);
    updateTransform();
  }
  /// \brief Copy constructor
  /// \param other **[in]** const reference to other Grid2 object
  Grid2(const Grid2 &other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = other.data_;
  }
  /// \brief Assign constructor
  /// \param other **[in]** temporary Grid2 object
  Grid2(Grid2 &&other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = std::move(other.data_);
  }
  /// \brief Constructs from raw data
  /// - Rows in ``values`` become values along the **x** axis.
  /// \param values **[in]** matrix data
  Grid2(Array2 <T> &values) { data_ = values; }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// Copy all fields from other object.
  /// \verbatim embed:rst:leading-slashes"
  ///   .. note::
  ///     Performs raw data copy.
  /// \endverbatim
  /// \param other **[in]** const reference to other Grid2 object
  Grid2 &operator=(const Grid2 &other) {
    to_grid_ = other.to_grid_;
    to_world_ = other.to_world_;
    origin_ = other.origin_;
    spacing_ = other.spacing_;
    data_ = other.data_;
    return *this;
  }
  /// Assign ``value`` to all cells
  /// \param value **[in]** assign value
  /// \return *this
  Grid2 &operator=(const T &value) {
    data_ = value;
    return *this;
  }
  /// Assign ``values`` to grid data
  /// \verbatim embed:rst:leading-slashes"
  ///   .. note::
  ///     The grid gets resized to match ``values`` dimensions.
  /// \endverbatim
  /// - Rows in ``values`` become values along the **x** axis.
  /// \param values **[in]** reference to matrix data
  /// \return *this
  Grid2 &operator=(Array2 <T> &values) {
    data_ = values;
    return *this;
  }
  /// Assign ``values`` to grid data
  /// \verbatim embed:rst:leading-slashes"
  ///   .. note::
  ///     The grid gets resized to match ``values`` dimensions.
  /// \endverbatim
  /// - Rows in ``values`` become values along the **x** axis.
  /// \param value **[in]** temporary Array2 object
  /// \return *this
  Grid2 &operator=(Array2 <T> &&values) {
    data_ = values;
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \return grid resolution (in cells count)
  size2 resolution() const { return data_.size(); }
  /// \return grid spacing (cell size)
  vec2 spacing() const { return spacing_; }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return origin_; }
  /// \param new_resolution **[in]** new grid resolution (in cells count)
  void setResolution(const size2 &new_resolution) {
    data_.resize(new_resolution);
  }
  /// \param new_spacing **[in]** new cell size
  void setSpacing(const vec2 &new_spacing) {
    spacing_ = new_spacing;
    updateTransform();
  }
  /// \param new_origin **[in]** new grid origin (world position of index (0,0))
  void setOrigin(const point2 &new_origin) {
    origin_ = new_origin;
    updateTransform();
  }
  /// Gets const raw data
  /// \return const reference to memory data
  const Array2 <T> &data() const { return data_; }
  /// Gets raw data
  /// \return reference to memory data
  Array2 <T> &data() { return data_; }
  /// \param address_mode **[in | default = CLAMP_TO_EDGE]** defines how out of
  /// bounds indices are handled
  /// \param interpolation_mode **[in | default = MONOTONIC_CUBIC]** what type
  /// of interpolation method is used on sampling
  /// \param border **[in | default = 0]** border value
  /// \return Grid2Accessor<T> accessor to grid data
  Grid2Accessor<T> accessor(
      AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
      InterpolationMode interpolation_mode = InterpolationMode::MONOTONIC_CUBIC,
      T border = T(0)) {
    return Grid2Accessor<T>(*this, address_mode, border, interpolation_mode);
  }
  /// \param address_mode **[in | default = CLAMP_TO_EDGE]** defines how out of
  /// bounds indices are handled
  /// \param interpolation_mode **[in | default = MONOTONIC_CUBIC]** what type
  /// of interpolation method is used on sampling
  /// \param border **[in | default = 0]** border value
  /// \return Grid2Accessor<T> accessor to grid data
  ConstGrid2Accessor<T> accessor(
      AddressMode address_mode = AddressMode::CLAMP_TO_EDGE,
      InterpolationMode interpolation_mode = InterpolationMode::MONOTONIC_CUBIC,
      T border = T(0)) const {
    return ConstGrid2Accessor<T>(*this, address_mode, border, interpolation_mode);
  }

private:
  void updateTransform() {
    to_world_ = translate(vec2(origin_.x, origin_.y)) * scale(spacing_);
    to_grid_ = inverse(to_world_);
  }
  Transform2 to_grid_, to_world_;
  point2 origin_;
  vec2 spacing_;
  Array2 <T> data_;
};

} // namespace ponos

#endif // PONOS_NUMERIC_GRID_H