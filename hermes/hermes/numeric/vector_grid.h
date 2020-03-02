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

#ifndef HERMES_NUMERIC_CUDA_VECTOR_GRID_H
#define HERMES_NUMERIC_CUDA_VECTOR_GRID_H

#include <hermes/numeric/grid.h>
#include <ponos/numeric/vector_grid.h>

namespace hermes {

namespace cuda {

// Accessor for arrays stored on the device.
// \tparam T data type
template <typename T> class VectorGrid2Accessor {
public:
  /// \param info **[in]**
  /// \param grid_type **[in]**
  /// \param u **[in]**
  /// \param v **[in]**
  VectorGrid2Accessor(Info2 info, ponos::VectorGridType grid_type,
                      Grid2Accessor<float> u, Grid2Accessor<float> v)
      : info_(info), grid_type_(grid_type), u_(u), v_(v) {}
  __host__ __device__ Grid2Accessor<T> &u() { return u_; }
  __host__ __device__ Grid2Accessor<T> &v() { return v_; }
  /// \return grid resolution
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
    return bbox2(wp, wp + info_.spacing());
  }
  /// \param ij **[in]**
  /// \return Vector2<T>
  __device__ Vector2<T> operator[](const index2 &ij) {
    switch (grid_type_) {
    case ponos::VectorGridType::CELL_CENTERED:
      return Vector2<T>(u_[ij], v_[ij]);
    case ponos::VectorGridType::STAGGERED:
      return Vector2<T>((u_[ij] + u_[ij.right()]) / 2,
                        (v_[ij] + v_[ij.up()]) / 2);
    }
    return Vector2<T>();
  }
  /// \param wp **[in]**
  /// \return Vector2<T>
  __device__ Vector2<T> operator()(const point2 &wp) {
    return {u_(wp), v_(wp)};
  }

protected:
  Info2 info_;
  ponos::VectorGridType grid_type_;
  Grid2Accessor<float> u_, v_;
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
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  VectorGrid2(
      ponos::VectorGridType grid_type = ponos::VectorGridType::CELL_CENTERED)
      : grid_type_(grid_type) {
    setSpacing(vec2f(1.f));
    setOrigin(point2f(0.f));
  }
  /// \param res resolution in number of cells
  /// \param s spacing cell size
  /// \param o origin (0,0,0) corner position
  VectorGrid2(
      size2 res, vec2f s, point2f o = point2f(),
      ponos::VectorGridType grid_type = ponos::VectorGridType::CELL_CENTERED)
      : grid_type_(grid_type) {
    setResolution(res);
    setSpacing(s);
    setOrigin(o);
  }
  /// \param host_grid **[in]**
  VectorGrid2(ponos::VectorGrid2<T> &host_grid) {
    setResolution(
        size2(host_grid.resolution().width, host_grid.resolution().height));
    setSpacing(vec2f(host_grid.spacing().x, host_grid.spacing().y));
    setOrigin(point2f(host_grid.origin().x, host_grid.origin().y));
    grid_type_ = host_grid.gridType();
    u_ = host_grid.u();
    v_ = host_grid.v();
  }
  VectorGrid2(const ponos::VectorGrid2<T> &host_grid) {
    setResolution(
        size2(host_grid.resolution().width, host_grid.resolution().height));
    setSpacing(vec2f(host_grid.spacing().x, host_grid.spacing().y));
    setOrigin(point2f(host_grid.origin().x, host_grid.origin().y));
    grid_type_ = host_grid.gridType();
    u_ = host_grid.u();
    v_ = host_grid.v();
  }
  VectorGrid2(const VectorGrid2<T> &other) {
    grid_type_ = other.grid_type_;
    setResolution(other.info_.resolution);
    setSpacing(other.info_.spacing());
    setOrigin(other.info_.origin());
    u_ = other.u_;
    v_ = other.v_;
  }
  VectorGrid2(VectorGrid2<T> &other) {
    grid_type_ = other.grid_type_;
    setResolution(other.info_.resolution);
    setSpacing(other.info_.spacing());
    setOrigin(other.info_.origin());
    u_ = other.u_;
    v_ = other.v_;
  }
  VectorGrid2(VectorGrid2<T> &&other) noexcept {
    grid_type_ = other.grid_type_;
    setResolution(other.info_.resolution);
    setSpacing(other.info_.spacing());
    setOrigin(other.info_.origin());
    u_ = std::move(other.u_);
    v_ = std::move(other.v_);
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  VectorGrid2 &operator=(const ponos::VectorGrid2<T> &host_grid) {
    setResolution(
        size2(host_grid.resolution().width, host_grid.resolution().height));
    setSpacing(vec2f(host_grid.spacing().x, host_grid.spacing().y));
    setOrigin(point2f(host_grid.origin().x, host_grid.origin().y));
    grid_type_ = host_grid.gridType();
    PING;
    u_ = host_grid.u();
    v_ = host_grid.v();
    return *this;
  }
  VectorGrid2 &operator=(ponos::VectorGrid2<T> &host_grid) {
    setResolution(
        size2(host_grid.resolution().width, host_grid.resolution().height));
    setSpacing(vec2f(host_grid.spacing().x, host_grid.spacing().y));
    setOrigin(point2f(host_grid.origin().x, host_grid.origin().y));
    grid_type_ = host_grid.gridType();
    u_ = host_grid.u();
    v_ = host_grid.v();
    return *this;
  }
  /// Copy all fields from other object. Performs raw data copy.
  /// \param other
  VectorGrid2 &operator=(const VectorGrid2 &other) {
    grid_type_ = other.grid_type_;
    setResolution(other.info_.resolution);
    setSpacing(other.info_.spacing());
    setOrigin(other.info_.origin());
    u_ = other.u_;
    v_ = other.v_;
    return *this;
  }
  /// Copy all fields from other object. Performs raw data copy.
  /// \param other
  VectorGrid2 &operator=(VectorGrid2 &other) {
    grid_type_ = other.grid_type_;
    setResolution(other.info_.resolution);
    setSpacing(other.info_.spacing());
    setOrigin(other.info_.origin());
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
  /// \param grid_type **[in]**
  void setGridType(ponos::VectorGridType grid_type) {
    if (grid_type != grid_type_) {
      grid_type_ = grid_type;
      setResolution(info_.resolution);
      setSpacing(info_.spacing());
      setOrigin(info_.origin());
    } else
      grid_type_ = grid_type;
  }
  /// \return VectorGridType
  ponos::VectorGridType gridType() const { return grid_type_; }

  /// \return size2 grid resolution
  size2 resolution() const { return info_.resolution; }
  /// \return vec2f grid spacing (cell size)
  vec2 spacing() const { return info_.spacing(); }
  /// \return grid origin (world position of index (0,0))
  point2 origin() const { return info_.origin(); }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void setResolution(size2 res) {
    info_.resolution = res;
    switch (grid_type_) {
    case ponos::VectorGridType::CELL_CENTERED:
      u_.setResolution(res);
      v_.setResolution(res);
      break;
    case ponos::VectorGridType::STAGGERED:
      u_.setResolution(res + size2(1, 0));
      v_.setResolution(res + size2(0, 1));
      break;
    default:
      break;
    }
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point2f &o) {
    info_.setOrigin(o);
    switch (grid_type_) {
    case ponos::VectorGridType::CELL_CENTERED:
      u_.setOrigin(o);
      v_.setOrigin(o);
      break;
    case ponos::VectorGridType::STAGGERED:
      u_.setOrigin(o - vec2(info_.spacing().x / 2, 0));
      v_.setOrigin(o - vec2(0, info_.spacing().y / 2));
      break;

    default:
      break;
    }
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setSpacing(const vec2f &s) {
    info_.setSpacing(s);
    u_.setSpacing(s);
    v_.setSpacing(s);
    setOrigin(info_.origin());
  }
  /// \return ponos::VectorGrid2<T>
  ponos::VectorGrid2<T> hostData() {
    ponos::VectorGrid2<T> h;
    h.setOrigin(info_.origin().ponos());
    h.setSpacing(info_.spacing().ponos());
    h.setResolution(info_.resolution.ponos());
    h.u() = u_.hostData();
    h.v() = v_.hostData();
    return h;
  }
  const Grid2<float> &u() const { return u_; }
  const Grid2<float> &v() const { return v_; }
  Grid2<float> &u() { return u_; }
  Grid2<float> &v() { return v_; }
  /// \param address_mode **[in]**
  /// \param interpolation_mode **[in]**
  /// \param border **[in]**
  /// \return VectorGrid2Accessor<T>
  VectorGrid2Accessor<T>
  accessor(ponos::AddressMode address_mode = ponos::AddressMode::CLAMP_TO_EDGE,
           ponos::InterpolationMode interpolation_mode =
               ponos::InterpolationMode::MONOTONIC_CUBIC,
           T border = T(0)) {
    return VectorGrid2Accessor<T>(
        info_, grid_type_,
        u_.accessor(address_mode, border, interpolation_mode),
        v_.accessor(address_mode, border, interpolation_mode));
  }

protected:
  Info2 info_;
  ponos::VectorGridType grid_type_{ponos::VectorGridType::CELL_CENTERED};
  Grid2<T> u_, v_;
};

// TODO: DEPRECATED
class VectorGridTexture2 {
public:
  VectorGridTexture2() {
    uGrid.setOrigin(point2f(0, 0));
    vGrid.setOrigin(point2f(0, 0));
  }
  /// \param resolution in number of cells
  /// \param origin (0,0) corner position
  /// \param dx cell size
  VectorGridTexture2(size2 resolution, point2f origin, float dx) {
    uGrid.resize(resolution);
    uGrid.setOrigin(origin);
    uGrid.setDx(dx);
    vGrid.resize(resolution);
    vGrid.setOrigin(origin);
    vGrid.setDx(dx);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(size2 res) {
    uGrid.resize(res);
    vGrid.resize(res);
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point2f &o) {
    uGrid.setOrigin(o);
    vGrid.setOrigin(o);
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setDx(float d) {
    uGrid.setDx(d);
    vGrid.setDx(d);
  }
  virtual void copy(const VectorGridTexture2 &other) {
    uGrid.copy(other.uGrid);
    vGrid.copy(other.vGrid);
  }
  virtual float *uDeviceData() { return uGrid.texture().deviceData(); }
  virtual float *vDeviceData() { return vGrid.texture().deviceData(); }
  virtual const float *uDeviceData() const {
    return uGrid.texture().deviceData();
  }
  virtual const float *vDeviceData() const {
    return vGrid.texture().deviceData();
  }
  virtual const GridTexture2<float> &u() const { return uGrid; }
  virtual const GridTexture2<float> &v() const { return vGrid; }
  virtual GridTexture2<float> &u() { return uGrid; }
  virtual GridTexture2<float> &v() { return vGrid; }

protected:
  GridTexture2<float> uGrid, vGrid;
};
// TODO: DEPRECATED
class VectorGridTexture3 {
public:
  VectorGridTexture3() {
    uGrid.setOrigin(point3f(0.0f));
    vGrid.setOrigin(point3f(0.0f));
    wGrid.setOrigin(point3f(0.0f));
  }
  /// \param resolution in number of cells
  /// \param origin (0,0,0) corner position
  /// \param dx cell size
  VectorGridTexture3(vec3u resolution, point3f origin, float dx) {
    uGrid.resize(resolution);
    uGrid.setOrigin(origin);
    uGrid.setDx(dx);
    vGrid.resize(resolution);
    vGrid.setOrigin(origin);
    vGrid.setDx(dx);
    wGrid.resize(resolution);
    wGrid.setOrigin(origin);
    wGrid.setDx(dx);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(vec3u res) {
    uGrid.resize(res);
    vGrid.resize(res);
    wGrid.resize(res);
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point3f &o) {
    uGrid.setOrigin(o);
    vGrid.setOrigin(o);
    wGrid.setOrigin(o);
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setDx(float d) {
    uGrid.setDx(d);
    vGrid.setDx(d);
    wGrid.setDx(d);
  }
  virtual void copy(const VectorGridTexture3 &other) {
    uGrid.copy(other.uGrid);
    vGrid.copy(other.vGrid);
    wGrid.copy(other.wGrid);
  }
  virtual float *uDeviceData() { return uGrid.texture().deviceData(); }
  virtual float *vDeviceData() { return vGrid.texture().deviceData(); }
  virtual float *wDeviceData() { return wGrid.texture().deviceData(); }
  virtual const float *uDeviceData() const {
    return uGrid.texture().deviceData();
  }
  virtual const float *vDeviceData() const {
    return vGrid.texture().deviceData();
  }
  virtual const float *wDeviceData() const {
    return wGrid.texture().deviceData();
  }
  virtual const GridTexture3<float> &u() const { return uGrid; }
  virtual const GridTexture3<float> &v() const { return vGrid; }
  virtual const GridTexture3<float> &w() const { return wGrid; }
  virtual GridTexture3<float> &u() { return uGrid; }
  virtual GridTexture3<float> &v() { return vGrid; }
  virtual GridTexture3<float> &w() { return wGrid; }

protected:
  GridTexture3<float> uGrid, vGrid, wGrid;
};

// template <MemoryLocation L>
// void fill3(VectorGrid3<L> &grid, const bbox3f &region, vec3f value,
//            bool overwrite = false);

// using VectorGrid2D = VectorGrid2<MemoryLocation::DEVICE>;
// using VectorGrid2H = VectorGrid2<MemoryLocation::HOST>;
// Accessor for arrays stored on the device.
// Indices are accessed as: i * width * height + j * height + k
// \tparam T data type
class VectorGrid3Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  VectorGrid3Accessor(const vec3u &resolution, const vec3f &spacing,
                      RegularGrid3Accessor<float> u,
                      RegularGrid3Accessor<float> v,
                      RegularGrid3Accessor<float> w)
      : resolution(resolution), spacing(spacing), u_(u), v_(v), w_(w) {}
  virtual __host__ __device__ vec3f operator()(int i, int j, int k) {
    return vec3f(u_(i, j, k), v_(i, j, k), w_(i, j, k));
  }
  virtual __host__ __device__ vec3f operator()(point3f wp) {
    return vec3f(u_(wp), v_(wp), w_(wp));
  }
  __host__ __device__ float &u(int i, int j, int k) { return u_(i, j, k); }
  __host__ __device__ float &v(int i, int j, int k) { return v_(i, j, k); }
  __host__ __device__ float &w(int i, int j, int k) { return w_(i, j, k); }
  __host__ __device__ float u(int i, int j, int k) const { return u_(i, j, k); }
  __host__ __device__ float v(int i, int j, int k) const { return v_(i, j, k); }
  __host__ __device__ float w(int i, int j, int k) const { return w_(i, j, k); }
  __host__ __device__ RegularGrid3Accessor<float> &uAccessor() { return u_; }
  __host__ __device__ RegularGrid3Accessor<float> &vAccessor() { return v_; }
  __host__ __device__ RegularGrid3Accessor<float> &wAccessor() { return w_; }

  const vec3u resolution;
  const vec3f spacing;

protected:
  RegularGrid3Accessor<float> u_, v_, w_;
};

template <MemoryLocation L> class VectorGrid3 {
public:
  VectorGrid3() {
    setSpacing(vec3f(1.f));
    setOrigin(point3f(0.f));
  }
  /// \param res resolution in number of cells
  /// \param o origin (0,0,0) corner position
  /// \param s spacing cell size
  VectorGrid3(vec3u res, point3f o, vec3f s)
      : resolution_(res), origin_(o), spacing_(s) {
    uGrid.resize(res);
    uGrid.setOrigin(o);
    uGrid.setSpacing(s);
    vGrid.resize(res);
    vGrid.setOrigin(o);
    vGrid.setSpacing(s);
    wGrid.resize(res);
    wGrid.setOrigin(o);
    wGrid.setSpacing(s);
  }
  vec3u resolution() const { return resolution_; }
  vec3f spacing() const { return spacing_; }
  point3f origin() const { return origin_; }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(vec3u res) {
    resolution_ = res;
    uGrid.resize(res);
    vGrid.resize(res);
    wGrid.resize(res);
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point3f &o) {
    origin_ = o;
    uGrid.setOrigin(o);
    vGrid.setOrigin(o);
    wGrid.setOrigin(o);
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setSpacing(const vec3f &s) {
    spacing_ = s;
    uGrid.setSpacing(s);
    vGrid.setSpacing(s);
    wGrid.setSpacing(s);
  }
  const RegularGrid3<L, float> &u() const { return uGrid; }
  const RegularGrid3<L, float> &v() const { return vGrid; }
  const RegularGrid3<L, float> &w() const { return wGrid; }
  RegularGrid3<L, float> &u() { return uGrid; }
  RegularGrid3<L, float> &v() { return vGrid; }
  RegularGrid3<L, float> &w() { return wGrid; }
  VectorGrid3Accessor accessor() {
    return VectorGrid3Accessor(resolution_, spacing_, uGrid.accessor(),
                               vGrid.accessor(), wGrid.accessor());
  }
  /// Copy data from other staggered grid
  /// other reference from other field
  template <MemoryLocation LL> void copy(VectorGrid3<LL> &other) {
    if (other.resolution() != resolution())
      resize(other.resolution());
    setOrigin(other.origin());
    setSpacing(other.spacing());
    memcpy(uGrid.data(), other.u().data());
    memcpy(vGrid.data(), other.v().data());
    memcpy(wGrid.data(), other.w().data());
  }

protected:
  vec3u resolution_;
  point3f origin_;
  vec3f spacing_;
  RegularGrid3<L, float> uGrid, vGrid, wGrid;
};

// template <MemoryLocation L>
// void fill3(VectorGrid3<L> &grid, const bbox3f &region, vec3f value,
//            bool overwrite = false);

using VectorGrid3D = VectorGrid3<MemoryLocation::DEVICE>;
using VectorGrid3H = VectorGrid3<MemoryLocation::HOST>;

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_VECTOR_GRID_H