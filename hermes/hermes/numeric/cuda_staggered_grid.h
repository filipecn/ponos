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

#ifndef HERMES_NUMERIC_CUDA_STAGGERED_GRID_H
#define HERMES_NUMERIC_CUDA_STAGGERED_GRID_H

#include <hermes/numeric/cuda_vector_field.h>
#include <hermes/storage/cuda_storage_utils.h>

namespace hermes {

namespace cuda {

/// Represents a staggered grid with texture grid
/// The origin of each staggered grid follows the scheme:
///    ----------
///   |         |
///   u    o    |
///   |         |
///   -----v----
class StaggeredGridTexture2 : public VectorGridTexture2 {
public:
  StaggeredGridTexture2() {
    uGrid.setOrigin(point2f(-0.5f, 0.0f));
    vGrid.setOrigin(point2f(0.0f, -0.5f));
  }
  /// \param resolution in number of cells
  /// \param origin (0,0) corner position
  /// \param dx cell size
  StaggeredGridTexture2(vec2u resolution, point2f origin, float dx) {
    uGrid.resize(resolution + vec2u(1, 0));
    uGrid.setOrigin(origin + vec2f(-0.5f, 0.f));
    uGrid.setDx(dx);
    vGrid.resize(resolution + vec2u(0, 1));
    vGrid.setOrigin(origin + vec2f(0.f, -0.5f));
    vGrid.setDx(dx);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(vec2u res) override {
    uGrid.resize(res + vec2u(1, 0));
    vGrid.resize(res + vec2u(0, 1));
  }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point2f &o) override {
    uGrid.setOrigin(o + vec2f(-0.5f, 0.f));
    vGrid.setOrigin(o + vec2f(0.f, -0.5f));
  }
};

class StaggeredGrid3Accessor : public VectorGrid3Accessor {
public:
  StaggeredGrid3Accessor(const vec3u &resolution, const vec3f &spacing,
                         RegularGrid3Accessor<float> u,
                         RegularGrid3Accessor<float> v,
                         RegularGrid3Accessor<float> w)
      : VectorGrid3Accessor(resolution, spacing, u, v, w) {}
  __host__ __device__ vec3f operator()(int i, int j, int k) override {
    return 0.5f * vec3f(this->u_(i + 1, j, k) + this->u_(i, j, k),
                        this->v_(i, j + 1, k) + this->v_(i, j, k),
                        this->w_(i, j, k + 1) + this->w_(i, j, k));
  }
};

/// Represents a staggered grid with regular grid
/// The origin of each staggered grid follows the scheme:
///    ----------
///   |         |
///   u    o    |
///   |  w      |
///   -----v----
template <MemoryLocation L> class StaggeredGrid3 : public VectorGrid3<L> {
public:
  StaggeredGrid3() {
    setSpacing(vec3f(1.f));
    setOrigin(point3f(0.f));
  }
  /// \param res resolution in number of cells
  /// \param o origin (0,0) corner position
  /// \param s cell size
  StaggeredGrid3(vec3u res, point3f o, const vec3f &s) {
    this->origin_ = o;
    this->resolution_ = res;
    this->spacing_ = s;
    this->uGrid.resize(res + vec3u(1, 0, 0));
    this->uGrid.setOrigin(o + s.x * vec3f(-0.5f, 0.f, 0.f));
    this->uGrid.setSpacing(s);
    this->vGrid.resize(res + vec3u(0, 1, 0));
    this->vGrid.setOrigin(o + s.y * vec3f(0.f, -0.5f, 0.f));
    this->vGrid.setSpacing(s);
    this->wGrid.resize(res + vec3u(0, 0, 1));
    this->wGrid.setOrigin(o + s.z * vec3f(0.f, 0.f, -0.5f));
    this->wGrid.setSpacing(s);
  }
  /// Changes grid cell size
  /// \param d new size
  void setSpacing(const vec3f &s) override {
    this->spacing_ = s;
    this->uGrid.setSpacing(s);
    this->vGrid.setSpacing(s);
    this->wGrid.setSpacing(s);
    setOrigin(this->origin_);
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(vec3u res) override {
    this->resolution_ = res;
    this->uGrid.resize(res + vec3u(1, 0, 0));
    this->vGrid.resize(res + vec3u(0, 1, 0));
    this->wGrid.resize(res + vec3u(0, 0, 1));
  }
  /// Changes grid origin position
  /// \param o in world space
  void setOrigin(const point3f &o) override {
    this->origin_ = o;
    this->uGrid.setOrigin(o + this->spacing_.x * vec3f(-0.5f, 0.f, 0.f));
    this->vGrid.setOrigin(o + this->spacing_.y * vec3f(0.f, -0.5f, 0.f));
    this->wGrid.setOrigin(o + this->spacing_.z * vec3f(0.f, 0.f, -0.5f));
  }
  StaggeredGrid3Accessor accessor() {
    return StaggeredGrid3Accessor(
        this->resolution_, this->spacing_, this->uGrid.accessor(),
        this->vGrid.accessor(), this->wGrid.accessor());
  }
};

using StaggeredGrid3D = StaggeredGrid3<MemoryLocation::DEVICE>;
using StaggeredGrid3H = StaggeredGrid3<MemoryLocation::HOST>;

} // namespace cuda

} // namespace hermes

#endif // HERMES_STRUCTURES_CUDA_STAGGERED_GRID_H