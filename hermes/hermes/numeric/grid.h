/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_NUMERIC_CUDA_GRID_H
#define HERMES_NUMERIC_CUDA_GRID_H

#include <hermes/numeric/cuda_field.h>
#include <hermes/numeric/cuda_interpolation.h>
#include <hermes/storage/array.h>
#include <ponos/numeric/grid.h>

namespace hermes {

namespace cuda {

class Info2 {
public:
  __host__ __device__ point2 toWorld(const point2 &gp) const {
    return to_world_(gp);
  }
  __host__ __device__ point2 toGrid(const point2 &wp) const {
    return to_grid_(wp);
  }
  __host__ __device__ const vec2 &spacing() const { return spacing_; }
  __host__ __device__ const point2 &origin() const { return origin_; }
  void set(const point2 &o, const vec2 &s) {
    origin_ = o;
    spacing_ = s;
    updateTransforms();
  }
  void setOrigin(const point2 &o) {
    origin_ = o;
    updateTransforms();
  }
  void setSpacing(const vec2 &s) {
    spacing_ = s;
    updateTransforms();
  }
  size2 resolution;

private:
  void updateTransforms() {
    to_world_ = translate(vec2f(origin_[0], origin_[1])) *
                scale(spacing_.x, spacing_.y);
    to_grid_ = inverse(to_world_);
  }
  Transform2<float> to_grid_;
  Transform2<float> to_world_;
  point2 origin_;
  vec2 spacing_;
};
/*****************************************************************************
*************************            GRID2           *************************
******************************************************************************/
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
    __device__ Element(T &v, const index2 &ij, const Grid2Accessor<T> &acc)
        : value(v), index_(ij), acc_(acc) {}
    /// \return current iteration grid index
    __device__ index2 index() const { return index_; }
    /// \return current iteration grid index's i component
    __device__ int i() const { return index_.i; }
    /// \return current iteration grid index's j component
    __device__ int j() const { return index_.j; }
    /// \return world position coordinates of the current index
    __device__ point2 worldPosition() const {
      return acc_.worldPosition(index_);
    }
    /// \return region (in world coordinates) of the grid cell respective to
    /// the current index
    __device__ bbox2 region() const { return acc_.cellRegion(index_); }
    /// Reference to the grid data stored in the current index
    T &value;

  private:
    index2 index_;
    const Grid2Accessor<T> &acc_;
  };
  ///
  /// \param grid_accessor grid to be iterated over
  /// \param ij starting grid index for iteration
  __device__ Grid2Iterator(Grid2Accessor<T> &grid_accessor, const index2 &ij)
      : acc_(grid_accessor),
        it(Index2Iterator<i32>(index2(0, 0),
                               index2(grid_accessor.resolution().width,
                                      grid_accessor.resolution().height),
                               ij)) {}
  /// \return increment operator to move to the next grid index
  __device__ Grid2Iterator &operator++() {
    ++it;
    return *this;
  }
  /// \return current grid index element
  __device__ Element operator*() { return Element(acc_[*it], *it, acc_); }
  /// operator required by the loop iteration code of c++
  /// \param other
  /// \return true if iterators are different
  __device__ bool operator!=(const Grid2Iterator<T> &other) {
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
  /// \param info
  /// \param address_mode
  /// \param border
  explicit __host__ Grid2Accessor(
      Info2 info, Array2Accessor<T> grid,
      ponos::AddressMode address_mode = ponos::AddressMode::CLAMP_TO_EDGE,
      T border = T(0),
      ponos::InterpolationMode interpolation_mode =
          ponos::InterpolationMode::MONOTONIC_CUBIC)
      : info_(info), grid_(grid), address_mode_(address_mode),
        interpolation_mode_(interpolation_mode), border_(border) {}
  /// \return grid resolutio),n
  __host__ __device__ size2 resolution() const { return info_.resolution; }
  /// \return grid spacing
  __host__ __device__ vec2 spacing() const { return info_.spacing(); }
  /// \return grid origin (world position of index (0,0))
  __host__ __device__ point2 origin() const { return info_.origin(); }
  /// \param world_position (in world coordinates)
  /// \return world_position in grid coordinates
  __host__ __device__ point2 gridPosition(const point2 &world_position) const {
    return info_.toGrid(world_position);
  }
  /// \param grid_position (in grid coordinates)
  /// \return grid_position in world coordinates
  __host__ __device__ point2 worldPosition(const point2 &grid_position) const {
    return info_.toWorld(grid_position);
  }
  /// \param grid_position (in grid coordinates)
  /// \return grid_position in world coordinates
  __host__ __device__ point2 worldPosition(const index2 &grid_position) const {
    return info_.toWorld(point2(grid_position.i, grid_position.j));
  }
  /// \param world_position (in world coordinates)
  /// \return offset of world_position inside the cell containing it (in [0,1]).
  __host__ __device__ point2 cellPosition(const point2 &world_position) const {
    auto gp = info_.toGrid(world_position);
    return gp - vec2(static_cast<int>(gp.x), static_cast<int>(gp.y));
  }
  /// \param world_position (in world coordinates)
  /// \return index of the cell that contains world_position
  __host__ __device__ index2 cellIndex(const point2 &world_position) const {
    auto gp = info_.toGrid(world_position);
    return index2(gp.x, gp.y);
  }
  /// \param ij cell index
  /// \return ij cell's region (in world coordinates)
  __host__ __device__ bbox2 cellRegion(const index2 &ij) const {
    auto wp = info_.toWorld(point2(ij.i, ij.j));
    return bbox2(wp, wp + grid_.spacing_);
  }
  __host__ __device__ bool stores(const index2 &ij) const {
    return info_.resolution.contains(ij.i, ij.j);
  }
  /// \param ij position index
  /// \return reference to data stored at ij based on the address mode
  __device__ T &operator[](const index2 &ij) {
    index2 fij = ij;
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      fij.i = (ij.i < 0) ? info_.resolution.width - 1 - ij.i
                         : ij.i % info_.resolution.width;
      fij.j = (ij.j < 0) ? info_.resolution.height - 1 - ij.j
                         : ij.j % info_.resolution.height;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      fij.clampTo(
          size2(info_.resolution.width - 1, info_.resolution.height - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (!grid_.contains(ij)) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }

    assert(grid_.contains(fij));
    return grid_[fij];
  }
  /// \param world_position (in world coordinates)
  /// \return sampled data value in world_position based on the interpolation
  /// mode and address mode
  __device__ T operator()(const point2 &world_position) {
    auto cp = cellPosition(world_position);
    auto ip = cellIndex(world_position);
    switch (interpolation_mode_) {
    case ponos::InterpolationMode::LINEAR:
      return bilerp<T>(cp.x, cp.y, (*this)[ip], (*this)[ip + index2(1, 0)],
                       (*this)[ip + index2(1, 1)], (*this)[ip + index2(0, 1)]);
    case ponos::InterpolationMode::MONOTONIC_CUBIC:
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
  __device__ Grid2Iterator<T> begin() {
    return Grid2Iterator<float>(*this, index2(0, 0));
  }
  /// \return sentinel iterator
  __device__ Grid2Iterator<T> end() {
    return Grid2Iterator<float>(
        *this, index2(info_.resolution.width, info_.resolution.height));
  }

private:
  Info2 info_;
  Array2Accessor<T> grid_;
  ponos::AddressMode address_mode_ = ponos::AddressMode::CLAMP_TO_EDGE;
  ponos::InterpolationMode interpolation_mode_ =
      ponos::InterpolationMode::LINEAR;
  T border_ = T(0);
  T dummy_ = T(0);
};
/// A Grid2 is a numerical discretization of an arbitrary 2-dimensional domain
/// that can be used for numerical simulations and other applications. Each
/// grid index represents the lower vertex of a grid cell, so the center of the
/// cell (i,j) is (i + .5, j + .5). In other words, data is stored at vertex
/// positions.
/// \tparam T grid data type
template <typename T> class Grid2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Grid must hold an float type!");

public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Grid2() = default;
  ///\brief Construct a new Grid2 object from other Grid2
  ///\param other **[in]**
  explicit Grid2(Grid2<T> &other) {
    info_ = other.info_;
    data_ = other.data_;
  }
  explicit Grid2(const Grid2<T> &other) {
    info_ = other.info_;
    data_ = other.data_;
  }
  ///\brief Construct a new Grid 2 object
  ///\param resolution **[in]** grid resolution
  ///\param spacing **[in]** cell size
  ///\param origin **[in]** bottom left vertex position
  Grid2(const size2 &resolution, const vec2 &spacing = vec2(1, 1),
        const point2 &origin = point2(0, 0)) {
    info_.resolution = resolution;
    info_.set(origin, spacing);
    data_.resize(resolution);
  }
  ///\brief Construct a new Grid2 object from host Grid2
  ///\param other **[in]** host grid
  explicit Grid2(ponos::Grid2<T> &other) {
    info_.resolution = size2(other.resolution());
    info_.set(point2(other.origin()), vec2(other.spacing()));
    data_ = other.data();
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// Copy host data to device
  ///\param other **[in]** host grid
  ///\return Grid2&
  Grid2 &operator=(ponos::Grid2<T> &other) {
    info_.resolution =
        size2(other.resolution().width, other.resolution().height);
    info_.set(point2f(other.origin().x, other.origin().y),
              vec2f(other.spacing().x, other.spacing().y));
    data_ = other.data();
    return *this;
  }
  /// Copy data
  ///\param other **[in]**
  ///\return Grid2&
  Grid2 &operator=(const Grid2 &other) {
    info_ = other.info_;
    data_ = other.data_;
    return *this;
  }
  /// Copy data
  ///\param other **[in]**
  ///\return Grid2&
  Grid2 &operator=(Grid2 &other) {
    info_ = other.info_;
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
  /// Assign raw data
  ///\param values **[in]** raw data
  ///\return Grid2&
  Grid2 &operator=(Array2<T> &values) {
    data_ = values;
    return *this;
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void setResolution(const size2 &res) {
    info_.resolution = res;
    data_.resize(res);
  }
  /// \return grid resolution
  size2 resolution() const { return info_.resolution; }
  /// \return grid spacing (cell size)
  vec2 spacing() const { return info_.spacing(); }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return info_.origin(); }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point2 &o) { info_.setOrigin(o); }
  /// Changes grid cell size
  /// \param s new size
  void setSpacing(const vec2 &s) { info_.setSpacing(s); }
  ///\return ponos::Grid2 host grid
  ponos::Grid2<T> hostData() {
    ponos::Grid2<T> g;
    g.setResolution(
        ponos::size2(info_.resolution.width, info_.resolution.height));
    g.setSpacing(info_.spacing().ponos());
    g.setOrigin(info_.origin().ponos());
    g = data_.hostData();
    return g;
  }
  ///\param addressMode **[in]**
  ///\param border **[in]**
  ///\return Grid2Accessor<T>
  Grid2Accessor<T>
  accessor(ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
           T border = T(0),
           ponos::InterpolationMode interpolation_mode =
               ponos::InterpolationMode::MONOTONIC_CUBIC) {
    return Grid2Accessor<T>(info_, data_.accessor(), addressMode, border,
                            interpolation_mode);
  }
  ///\return Array2<T>& raw data
  Array2<T> &data() { return data_; }
  ///\return Array2<T>& const raw data
  const Array2<T> &data() const { return data_; }
  ///\return const Info2& grid information
  const Info2 &info() const { return info_; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  ///\tparam F function type
  ///\param operation **[in]**
  template <typename F> void map(F operation) { data_.map(operation); }

private:
  Info2 info_{};
  Array2<T> data_;
};

// TODO: DEPRECATED
struct Grid2Info {
  Transform2<float> toField;
  Transform2<float> toWorld;
  size2 resolution;
  point2f origin;
  float dx;
};

// TODO: DEPRECATED
struct Grid3Info {
  Transform<float> toField;
  Transform<float> toWorld;
  vec3u resolution;
  point3f origin;
  float dx;
};

struct RegularGrid2Info {
  Transform2<float> toGrid;
  Transform2<float> toWorld;
  size2 resolution;
  point2f origin;
  vec2f spacing;
};

struct RegularGrid3Info {
  Transform<float> toGrid;
  Transform<float> toWorld;
  vec3u resolution;
  point3f origin;
  vec3f spacing;
};
/*
template <typename T> class RegularGrid2Iterator {
public:
  class Element {
  public:
    __host__ __device__ Element(T &v, const vec2i &ij, RegularGrid2Info &info)
        : value(v), index_(ij), info_(info) {}
    __host__ __device__ vec2i index() const { return index_; }
    __host__ __device__ int i() const { return index_.x; }
    __host__ __device__ int j() const { return index_.y; }
    __host__ __device__ point2f worldPosition() const {
      return info_.toWorld(point2f(index_.x, index_.y));
    }

    T &value;

  private:
    vec2i index_;
    RegularGrid2Info &info_;
  };
  __host__ __device__ RegularGrid2Iterator(Array2Accessor<T> &dataAccessor,
                                           RegularGrid2Info &info,
                                           const vec2i &ij)
      : acc_(dataAccessor), info_(info), size_(info.resolution), i(ij.x),
        j(ij.y) {}
  __host__ __device__ size2 size() const { return size_; }
  __host__ __device__ RegularGrid2Iterator &operator++() {
    i++;
    if (i >= size_.x) {
      i = 0;
      j++;
      if (j >= size_.y) {
        i = j = -1;
      }
    }
    return *this;
  }
  __host__ __device__ Element operator*() {
    return Element(acc_(i, j), vec2i(i, j), info_);
  }
  __host__ __device__ bool operator!=(const RegularGrid2Iterator &other) {
    return size_ != other.size_ || i != other.i || j != other.j;
  }

private:
  Array2Accessor<T> acc_;
  RegularGrid2Info info_;
  int i = 0, j = 0;
  size2 size_;
};*/
/*
/// Accessor for arrays stored on the device.
/// \tparam T data type
template <typename T> class RegularGrid2Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid2Accessor(
      const RegularGrid2Info &info, Array2Accessor<T> data,
      ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
      T border = T(0))
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ size2 resolution() const { return data_.size(); }
  __host__ __device__ vec2f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ T &operator()(int i, int j) {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    if (!data_.isIndexValid(i, j))
      printf("WARNING: Accessing invalid index from RegularGrid2Accessor!\n");
    return data_(i, j);
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \return const T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ const T &operator()(int i, int j) const {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y)
        return border_;
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j);
  }
  __host__ __device__ point2f worldPosition(int i, int j) {
    return info_.toWorld(point2f(i, j));
  }
  __host__ __device__ point2f gridPosition(const point2f &wp) {
    return info_.toGrid(wp);
  }
  ///
  __host__ __device__ bool isIndexStored(int i, int j) {
    return i >= 0 && i < data_.size().x && j >= 0 && j < data_.size().y;
  }
  __host__ __device__ RegularGrid2Iterator<T> begin() {
    return RegularGrid2Iterator<T>(data_, info_, vec2i(0));
  }
  __host__ __device__ RegularGrid2Iterator<T> end() {
    return RegularGrid2Iterator<T>(data_, info_, vec2i(-1));
  }

private:
  RegularGrid2Info info_;
  Array2Accessor<T> data_;
  ponos::AddressMode
      address_mode_; //!< defines how out of bounds data is treated
  T border_;         //!< border value
  T dummy_;          //!< used as out of bounds reference variable
};

template <> class RegularGrid2Accessor<float> {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid2Accessor(
      const RegularGrid2Info &info, Array2Accessor<float> data,
      ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
      float border = 0.f)
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ size2 resolution() const { return data_.size(); }
  __host__ __device__ vec2f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ float &operator()(int i, int j) {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j);
  }
  // sample
  __host__ __device__ float operator()(const point2f &wp) {
    auto gp = info_.toGrid(wp);
    int i = gp.x;
    int j = gp.y;
    float f[4][4];
    for (int dj = -1, J = 0; dj <= 2; dj++, J++)
      for (int di = -1, I = 0; di <= 2; di++, I++)
        f[J][I] = (*this)(i + di, j + dj);
    return monotonicCubicInterpolate(f, gp);
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return const T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ const float &operator()(int i, int j) const {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y)
        return border_;
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j);
  }
  __host__ __device__ point2f worldPosition(int i, int j) {
    return info_.toWorld(point2f(i, j));
  }
  __host__ __device__ point2f gridPosition(const point2f &wp) {
    return info_.toGrid(wp);
  }
  ///
  __host__ __device__ bool isIndexStored(int i, int j) {
    return i >= 0 && i < data_.size().x && j >= 0 && j < data_.size().y;
  }
  __host__ __device__ RegularGrid2Iterator<float> begin() {
    return RegularGrid2Iterator<float>(data_, info_, vec2i(0));
  }
  __host__ __device__ RegularGrid2Iterator<float> end() {
    return RegularGrid2Iterator<float>(data_, info_, vec2i(-1));
  }

private:
  RegularGrid2Info info_;
  Array2Accessor<float> data_;
  ponos::AddressMode
      address_mode_; //!< defines how out of bounds data is treated
  float border_;     //!< border value
  float dummy_;      //!< used as out of bounds reference variable
};                   // namespace cuda

/// Represents a regular grid that can be used in numeric calculations
template <MemoryLocation L, typename T> class RegularGrid2 {
public:
  template <MemoryLocation LL> RegularGrid2(RegularGrid2<LL, T> &other) {
    copy(other);
  }
  RegularGrid2(const size2 &size = size2()) {
    info_.resolution = size;
    data_.resize(size);
    if (size.x * size.y != 0)
      data_.allocate();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(const size2 &res) {
    info_.resolution = res;
    data_.resize(res);
    data_.allocate();
  }
  size2 resolution() const { return info_.resolution; }
  vec2f spacing() const { return info_.spacing; }
  point2f origin() const { return info_.origin; }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point2f &o) {
    info_.origin = o;
    updateTransform();
  }
  /// Changes grid cell size
  /// \param s new size
  void setSpacing(const vec2f &s) {
    info_.spacing = s;
    updateTransform();
  }
  RegularGrid2Accessor<T>
  accessor(ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
           T border = T(0)) {
    return RegularGrid2Accessor<T>(info_, data_.accessor(), addressMode,
                                   border);
  }
  Array2<L, T> &data() { return data_; }
  const Array2<L, T> &data() const { return data_; }
  const RegularGrid2Info &info() const { return info_; }
  template <MemoryLocation LL> void copy(RegularGrid2<LL, T> &other) {
    info_ = other.info();
    data_.resize(other.data().size());
    data_.allocate();
    memcpy(data_, other.data());
  }

private:
  void updateTransform() {
    info_.toWorld = translate(vec2f(info_.origin[0], info_.origin[1])) *
                    scale(info_.spacing.x, info_.spacing.y);
    info_.toGrid = inverse(info_.toWorld);
  }

  RegularGrid2Info info_;
  Array2<L, T> data_;
};

using RegularGrid2Df = RegularGrid2<MemoryLocation::DEVICE, float>;
using RegularGrid2Duc = RegularGrid2<MemoryLocation::DEVICE, unsigned char>;
using RegularGrid2Di = RegularGrid2<MemoryLocation::DEVICE, int>;
using RegularGrid2Hf = RegularGrid2<MemoryLocation::HOST, float>;
using RegularGrid2Huc = RegularGrid2<MemoryLocation::HOST, unsigned char>;
using RegularGrid2Hi = RegularGrid2<MemoryLocation::HOST, int>;
*/
template <typename T> class RegularGrid3Iterator {
public:
  class Element {
  public:
    __host__ __device__ Element(T &v, const vec3i &ij, RegularGrid3Info &info)
        : value(v), index_(ij), info_(info) {}
    __host__ __device__ vec3i index() const { return index_; }
    __host__ __device__ int i() const { return index_.x; }
    __host__ __device__ int j() const { return index_.y; }
    __host__ __device__ int k() const { return index_.z; }
    __host__ __device__ point3f worldPosition() const {
      return info_.toWorld(point3f(index_.x, index_.y, index_.z));
    }

    T &value;

  private:
    vec3i index_;
    RegularGrid3Info &info_;
  };
  __host__ __device__
  RegularGrid3Iterator(MemoryBlock3Accessor<T> &dataAccessor,
                       RegularGrid3Info &info, const vec3i &ijk)
      : acc_(dataAccessor), info_(info), size_(info.resolution), i(ijk.x),
        j(ijk.y), k(ijk.z) {}
  __host__ __device__ vec3u size() const { return size_; }
  __host__ __device__ RegularGrid3Iterator &operator++() {
    i++;
    if (i >= size_.x) {
      i = 0;
      j++;
      if (j >= size_.y) {
        i = 0;
        j = 0;
        k++;
        if (k >= size_.z)
          i = j = k = -1;
      }
    }
    return *this;
  }
  __host__ __device__ Element operator*() {
    return Element(acc_(i, j, k), vec3i(i, j, k), info_);
  }
  __host__ __device__ bool operator!=(const RegularGrid3Iterator &other) {
    return size_ != other.size_ || i != other.i || j != other.j;
  }

private:
  MemoryBlock3Accessor<T> acc_;
  RegularGrid3Info info_;
  int i = 0, j = 0, k = 0;
  vec3u size_;
};
/// Accessor for arrays stored on the device.
/// Indices are accessed as: i * width * height + j * height + k
/// \tparam T data type
template <typename T> class RegularGrid3Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid3Accessor(
      const RegularGrid3Info &info, MemoryBlock3Accessor<T> data,
      ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
      T border = T(0))
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ vec3u resolution() const { return data_.size(); }
  __host__ __device__ vec3f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ T &operator()(int i, int j, int k) {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j, k);
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return const T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ const T &operator()(int i, int j, int k) const {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z)
        return border_;
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j, k);
  }
  __host__ __device__ point3f worldPosition(int i, int j, int k) {
    return info_.toWorld(point3f(i, j, k));
  }
  __host__ __device__ point3f gridPosition(const point3f &wp) {
    return info_.toGrid(wp);
  }
  ///
  __host__ __device__ bool isIndexStored(int i, int j, int k) {
    return i >= 0 && i < data_.size().x && j >= 0 && j < data_.size().y &&
           k >= 0 && k < data_.size().z;
  }
  __host__ __device__ RegularGrid3Iterator<T> begin() {
    return RegularGrid3Iterator<T>(data_, info_, vec3i(0));
  }
  __host__ __device__ RegularGrid3Iterator<T> end() {
    return RegularGrid3Iterator<T>(data_, info_, vec3i(-1));
  }

private:
  RegularGrid3Info info_;
  MemoryBlock3Accessor<T> data_;
  ponos::AddressMode
      address_mode_; //!< defines how out of bounds data is treated
  T border_;         //!< border value
  T dummy_;          //!< used as out of bounds reference variable
};

template <> class RegularGrid3Accessor<float> {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid3Accessor(
      const RegularGrid3Info &info, MemoryBlock3Accessor<float> data,
      ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
      float border = 0.f)
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ vec3u resolution() const { return data_.size(); }
  __host__ __device__ vec3f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ float &operator()(int i, int j, int k) {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z) {
        dummy_ = border_;
        return dummy_;
      }
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j, k);
  }
  // sample
  __host__ __device__ float operator()(const point3f &wp) {
    auto gp = info_.toGrid(wp);
    int i = gp.x;
    int j = gp.y;
    int k = gp.z;
    // if (i == 15 && j == 15 && k == 13)
    //   printf("gp %f %f %f\n", gp.x, gp.y, gp.z);
    // std::cerr << "GP " << gp;
    // std::cerr << "ijk " << i << " " << j << " " << k << std::endl;
    float f[4][4][4];
    for (int dk = -1, K = 0; dk <= 2; dk++, K++)
      for (int dj = -1, J = 0; dj <= 2; dj++, J++)
        for (int di = -1, I = 0; di <= 2; di++, I++) {
          f[K][J][I] = (*this)(i + di, j + dj, k + dk);
          // if (i == 15 && j == 15 && k == 13) {
          //   printf("%d %d %d = %f\n", i + di, j + dj, k + dk, f[K][J][I]);
          // }
        }
    return monotonicCubicInterpolate(f, gp);
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return const T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ const float &operator()(int i, int j, int k) const {
    switch (address_mode_) {
    case ponos::AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case ponos::AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case ponos::AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z)
        return border_;
      break;
    case ponos::AddressMode::WRAP:
      break;
    case ponos::AddressMode::MIRROR:
      break;
    default:
      break;
    }
    return data_(i, j, k);
  }
  __host__ __device__ point3f worldPosition(int i, int j, int k) {
    return info_.toWorld(point3f(i, j, k));
  }
  __host__ __device__ point3f gridPosition(const point3f &wp) {
    return info_.toGrid(wp);
  }
  ///
  __host__ __device__ bool isIndexStored(int i, int j, int k) {
    return i >= 0 && i < data_.size().x && j >= 0 && j < data_.size().y &&
           k >= 0 && k < data_.size().z;
  }
  __host__ __device__ RegularGrid3Iterator<float> begin() {
    return RegularGrid3Iterator<float>(data_, info_, vec3i(0));
  }
  __host__ __device__ RegularGrid3Iterator<float> end() {
    return RegularGrid3Iterator<float>(data_, info_, vec3i(-1));
  }

private:
  RegularGrid3Info info_;
  MemoryBlock3Accessor<float> data_;
  ponos::AddressMode
      address_mode_; //!< defines how out of bounds data is treated
  float border_;     //!< border value
  float dummy_;      //!< used as out of bounds reference variable
};

/// Represents a regular grid that can be used in numeric calculations
template <MemoryLocation L, typename T> class RegularGrid3 {
public:
  template <MemoryLocation LL> RegularGrid3(RegularGrid3<LL, T> &other) {
    copy(other);
  }
  RegularGrid3(const vec3u &size = vec3u()) {
    info_.resolution = size;
    data_.resize(size);
    if (size.x * size.y * size.z != 0)
      data_.allocate();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(const vec3u &res) {
    info_.resolution = res;
    data_.resize(res);
    data_.allocate();
  }
  vec3u resolution() const { return info_.resolution; }
  vec3f spacing() const { return info_.spacing; }
  point3f origin() const { return info_.origin; }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point3f &o) {
    info_.origin = o;
    updateTransform();
  }
  /// Changes grid cell size
  /// \param s new size
  void setSpacing(const vec3f &s) {
    info_.spacing = s;
    updateTransform();
  }
  RegularGrid3Accessor<T>
  accessor(ponos::AddressMode addressMode = ponos::AddressMode::CLAMP_TO_EDGE,
           T border = T(0)) {
    return RegularGrid3Accessor<T>(info_, data_.accessor(), addressMode,
                                   border);
  }
  MemoryBlock3<L, T> &data() { return data_; }
  const MemoryBlock3<L, T> &data() const { return data_; }
  const RegularGrid3Info &info() const { return info_; }
  template <MemoryLocation LL> void copy(RegularGrid3<LL, T> &other) {
    info_ = other.info();
    data_.resize(other.data().size());
    data_.allocate();
    memcpy(data_, other.data());
  }

private:
  void updateTransform() {
    info_.toWorld =
        translate(vec3f(info_.origin[0], info_.origin[1], info_.origin[2])) *
        scale(info_.spacing.x, info_.spacing.y, info_.spacing.z);
    info_.toGrid = inverse(info_.toWorld);
  }

  RegularGrid3Info info_;
  MemoryBlock3<L, T> data_;
};

using RegularGrid3Df = RegularGrid3<MemoryLocation::DEVICE, float>;
using RegularGrid3Duc = RegularGrid3<MemoryLocation::DEVICE, unsigned char>;
using RegularGrid3Di = RegularGrid3<MemoryLocation::DEVICE, int>;
using RegularGrid3Hf = RegularGrid3<MemoryLocation::HOST, float>;
using RegularGrid3Huc = RegularGrid3<MemoryLocation::HOST, unsigned char>;
using RegularGrid3Hi = RegularGrid3<MemoryLocation::HOST, int>;

template <MemoryLocation L, typename T>
void fill3(RegularGrid3<L, T> &grid, const bbox3f &region, T value,
           bool overwrite = false);

template <typename T>
__global__ void __fill3(RegularGrid3Accessor<T> acc, bbox3f region, T value,
                        bool increment) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (acc.isIndexStored(x, y, z)) {
    auto wp = acc.worldPosition(x, y, z);
    if (region.contains(wp)) {
      if (increment)
        acc(x, y, z) += value;
      else
        acc(x, y, z) = value;
    }
  }
}

template <typename T>
void fill3(RegularGrid3<MemoryLocation::DEVICE, T> &grid, const bbox3f &region,
           T value, bool increment = false) {
  ThreadArrayDistributionInfo td(grid.resolution());
  __fill3<<<td.gridSize, td.blockSize>>>(grid.accessor(), region, value,
                                         increment);
}
/*
template <MemoryLocation L, typename T> T minValue(RegularGrid2<L, T> &grid);

template <typename T>
T minValue(RegularGrid2<MemoryLocation::DEVICE, T> &grid) {
  return minValue(grid.data());
}
*/
// template <MemoryLocation L, typename T> T maxValue(RegularGrid2<L, T> &grid);
/*
template <typename T>
T maxValue(RegularGrid2<MemoryLocation::DEVICE, T> &grid) {
  return maxValue(grid.data());
}
*/

// TODO: DEPRECATED
/// Represents a texture field with position offset and scale
template <typename T> class GridTexture2 {
public:
  GridTexture2() = default;
  /// \param resolution in number of cells
  /// \param origin (0,0) coordinate position
  /// \param dx cell size
  GridTexture2(size2 resolution, point2f origin, float dx)
      : origin(origin), dx(dx) {
    texGrid.resize(resolution);
    updateTransform();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(size2 res) { texGrid.resize(res); }
  size2 resolution() const { return texGrid.resolution(); }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point2f &o) {
    origin = o;
    updateTransform();
  }
  /// Changes grid cell size
  /// \param d new size
  void setDx(float d) {
    dx = d;
    updateTransform();
  }
  T minValue() {}
  /// \return Info grid info to be passed to kernels
  Grid2Info info() const {
    return {texGrid.toFieldTransform(), texGrid.toWorldTransform(),
            size2(texGrid.texture().width(), texGrid.texture().height()),
            origin, dx};
  }
  void copy(const GridTexture2 &other) {
    origin = other.origin;
    dx = other.dx;
    texGrid.texture().copy(other.texGrid.texture());
  }
  ///
  /// \return Texture<T>&
  Texture<T> &texture() { return texGrid.texture(); }
  ///
  /// \return const Texture<T>&
  const Texture<T> &texture() const { return texGrid.texture(); }
  ///
  /// \return Transform<float, 2>
  Transform2<float> toWorldTransform() const {
    return texGrid.toWorldTransform();
  }
  ///
  /// \return Transform2<float>
  Transform2<float> toFieldTransform() const {
    return texGrid.toFieldTransform();
  }

private:
  void updateTransform() {
    texGrid.setTransform(scale(dx, dx) *
                         translate(vec2f(origin[0], origin[1])));
  }
  point2f origin;
  float dx = 1.f;
  FieldTexture2<T> texGrid;
};

// TODO: DEPRECATED
/// Represents a texture field with position offset and scale
template <typename T> class GridTexture3 {
public:
  GridTexture3() = default;
  /// \param resolution in number of cells
  /// \param origin (0,0) coordinate position
  /// \param dx cell size
  GridTexture3(vec3u resolution, point3f origin, float dx)
      : origin(origin), dx(dx) {
    texGrid.resize(resolution);
    updateTransform();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(vec3u res) { texGrid.resize(res); }
  vec3u resolution() const { return texGrid.resolution(); }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point3f &o) {
    origin = o;
    updateTransform();
  }
  /// Changes grid cell size
  /// \param d new size
  void setDx(float d) {
    dx = d;
    updateTransform();
  }
  T minValue() {}
  /// \return Info grid info to be passed to kernels
  Grid3Info info() const {
    return {texGrid.toFieldTransform(), texGrid.toWorldTransform(),
            vec3u(texGrid.texture().width(), texGrid.texture().height(),
                  texGrid.texture().depth()),
            origin, dx};
  }
  void copy(const GridTexture3 &other) {
    origin = other.origin;
    dx = other.dx;
    texGrid.texture().copy(other.texGrid.texture());
  }
  ///
  /// \return Texture<T>&
  Texture3<T> &texture() { return texGrid.texture(); }
  ///
  /// \return const Texture<T>&
  const Texture3<T> &texture() const { return texGrid.texture(); }
  ///
  /// \return Transform<float, 3>
  Transform<float> toWorldTransform() const {
    return texGrid.toWorldTransform();
  }
  ///
  /// \return Transform3<float>
  Transform<float> toFieldTransform() const {
    return texGrid.toFieldTransform();
  }

private:
  void updateTransform() {
    texGrid.setTransform(scale(dx, dx, dx) *
                         translate(vec3f(origin[0], origin[1], origin[2])));
    texGrid.setTransform(translate(vec3f(origin[0], origin[1], origin[2])) *
                         scale(dx, dx, dx));
  }
  point3f origin;
  float dx = 1.f;
  FieldTexture3<T> texGrid;
};
} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_GRID_H