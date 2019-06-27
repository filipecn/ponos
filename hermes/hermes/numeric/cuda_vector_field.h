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

#include <hermes/numeric/cuda_grid.h>

namespace hermes {

namespace cuda {

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
  VectorGridTexture2(vec2u resolution, point2f origin, float dx) {
    uGrid.resize(resolution);
    uGrid.setOrigin(origin);
    uGrid.setDx(dx);
    vGrid.resize(resolution);
    vGrid.setOrigin(origin);
    vGrid.setDx(dx);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(vec2u res) {
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

// Accessor for arrays stored on the device.
// \tparam T data type
class VectorGrid2Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  VectorGrid2Accessor(const vec2u &resolution, const vec2f &spacing,
                      RegularGrid2Accessor<float> u,
                      RegularGrid2Accessor<float> v)
      : resolution(resolution), spacing(spacing), u_(u), v_(v) {}
  virtual __host__ __device__ vec2f operator()(int i, int j) {
    return vec2f(u_(i, j), v_(i, j));
  }
  virtual __host__ __device__ vec2f operator()(point2f wp) {
    return vec2f(u_(wp), v_(wp));
  }
  __host__ __device__ float &u(int i, int j) { return u_(i, j); }
  __host__ __device__ float &v(int i, int j) { return v_(i, j); }
  __host__ __device__ float u(int i, int j) const { return u_(i, j); }
  __host__ __device__ float v(int i, int j) const { return v_(i, j); }
  __host__ __device__ RegularGrid2Accessor<float> &uAccessor() { return u_; }
  __host__ __device__ RegularGrid2Accessor<float> &vAccessor() { return v_; }

  const vec2u resolution;
  const vec2f spacing;

protected:
  RegularGrid2Accessor<float> u_, v_;
};

template <MemoryLocation L> class VectorGrid2 {
public:
  VectorGrid2() {
    setSpacing(vec2f(1.f));
    setOrigin(point2f(0.f));
  }
  /// \param res resolution in number of cells
  /// \param o origin (0,0,0) corner position
  /// \param s spacing cell size
  VectorGrid2(vec2u res, point2f o, vec2f s)
      : resolution_(res), origin_(o), spacing_(s) {
    uGrid.resize(res);
    uGrid.setOrigin(o);
    uGrid.setSpacing(s);
    vGrid.resize(res);
    vGrid.setOrigin(o);
    vGrid.setSpacing(s);
  }
  vec2u resolution() const { return resolution_; }
  vec2f spacing() const { return spacing_; }
  point2f origin() const { return origin_; }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  virtual void resize(vec2u res) {
    resolution_ = res;
    uGrid.resize(res);
    vGrid.resize(res);
  }
  /// Changes grid origin position
  /// \param o in world space
  virtual void setOrigin(const point2f &o) {
    origin_ = o;
    uGrid.setOrigin(o);
    vGrid.setOrigin(o);
  }
  /// Changes grid cell size
  /// \param d new size
  virtual void setSpacing(const vec2f &s) {
    spacing_ = s;
    uGrid.setSpacing(s);
    vGrid.setSpacing(s);
  }
  const RegularGrid2<L, float> &u() const { return uGrid; }
  const RegularGrid2<L, float> &v() const { return vGrid; }
  RegularGrid2<L, float> &u() { return uGrid; }
  RegularGrid2<L, float> &v() { return vGrid; }
  VectorGrid2Accessor accessor() {
    return VectorGrid2Accessor(resolution_, spacing_, uGrid.accessor(),
                               vGrid.accessor());
  }
  /// Copy data from other staggered grid
  /// other reference from other field
  template <MemoryLocation LL> void copy(VectorGrid2<LL> &other) {
    if (other.resolution() != resolution())
      resize(other.resolution());
    setOrigin(other.origin());
    setSpacing(other.spacing());
    memcpy(uGrid.data(), other.u().data());
    memcpy(vGrid.data(), other.v().data());
  }

protected:
  vec2u resolution_;
  point2f origin_;
  vec2f spacing_;
  RegularGrid2<L, float> uGrid, vGrid;
};

// template <MemoryLocation L>
// void fill3(VectorGrid3<L> &grid, const bbox3f &region, vec3f value,
//            bool overwrite = false);

using VectorGrid2D = VectorGrid2<MemoryLocation::DEVICE>;
using VectorGrid2H = VectorGrid2<MemoryLocation::HOST>;
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