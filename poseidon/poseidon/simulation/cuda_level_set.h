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

#ifndef POSEIDON_SIMULATION_CUDA_LEVEL_SET_H
#define POSEIDON_SIMULATION_CUDA_LEVEL_SET_H

#include <hermes/algorithms/cuda_marching_cubes.h>
#include <hermes/numeric/cuda_grid.h>
#include <hermes/numeric/cuda_numeric.h>

namespace poseidon {

namespace cuda {

class SDF {
public:
  static __device__ float box(const hermes::cuda::bbox2f &box,
                              const hermes::cuda::point2f &p) {
    if (box.contains(p))
      return fmaxf(box.lower.x - p.x,
                   fmaxf(p.x - box.upper.x,
                         fmaxf(box.lower.y - p.y, p.y - box.upper.y)));
    hermes::cuda::point2f c = p;
    if (p.x < box.lower.x)
      c.x = box.lower.x;
    else if (p.x > box.upper.x)
      c.x = box.upper.x;
    if (p.y < box.lower.y)
      c.y = box.lower.y;
    else if (p.y > box.upper.y)
      c.y = box.upper.y;
    return distance(c, p);
  }
  static __device__ float box(const hermes::cuda::bbox3f &box,
                              const hermes::cuda::point3f &p) {
    if (box.contains(p))
      return fmaxf(
          box.lower.x - p.x,
          fmaxf(p.x - box.upper.x,
                fmaxf(box.lower.y - p.y,
                      fmaxf(p.y - box.upper.y,
                            fmaxf(box.lower.z - p.z, p.z - box.upper.z)))));
    hermes::cuda::point3f c = p;
    if (p.x < box.lower.x)
      c.x = box.lower.x;
    else if (p.x > box.upper.x)
      c.x = box.upper.x;
    if (p.y < box.lower.y)
      c.y = box.lower.y;
    else if (p.y > box.upper.y)
      c.y = box.upper.y;
    if (p.z < box.lower.z)
      c.z = box.lower.z;
    else if (p.z > box.upper.z)
      c.z = box.upper.z;
    return distance(c, p);
  }
};

/// Auxiliar class to access the level set properties
class LevelSet2Accessor {
public:
  LevelSet2Accessor(
      const hermes::cuda::RegularGrid2Accessor<float> &grid_accessor)
      : acc_(grid_accessor) {}
  __host__ __device__ hermes::cuda::vec2u resolution() const {
    return acc_.resolution();
  }
  __host__ __device__ hermes::cuda::vec2f spacing() const {
    return acc_.spacing();
  }
  __host__ __device__ float &operator()(int i, int j) { return acc_(i, j); }
  __host__ __device__ const float &operator()(int i, int j) const {
    return acc_(i, j);
  }
  __host__ __device__ hermes::cuda::vec2f gradient(int i, int j) const {
    return gradientAt(acc_, i, j, 1);
  }
  __host__ __device__ hermes::cuda::point2f worldPosition(int i, int j) {
    return acc_.worldPosition(i, j);
  }
  __host__ __device__ hermes::cuda::point2f
  gridPosition(const hermes::cuda::point2f &wp) {
    return acc_.gridPosition(wp);
  }
  __host__ __device__ bool isIndexStored(int i, int j) {
    return acc_.isIndexStored(i, j);
  }

private:
  hermes::cuda::RegularGrid2Accessor<float> acc_;
};

/// Represent a field of distances from a isoline
/// \tparam L memory location
template <hermes::cuda::MemoryLocation L> class LevelSet2 {
public:
  LevelSet2() {}
  /// \brief Construct a new Level Set 2 object
  /// \param resolution regular grid resolution
  /// \param spacing point spacing
  /// \param origin offset
  LevelSet2(const hermes::cuda::vec2u &resolution,
            const hermes::cuda::vec2f &spacing,
            const hermes::cuda::point2f &origin = hermes::cuda::point2f(0.f)) {
    grid_.resize(resolution);
    grid_.setOrigin(origin);
    grid_.setSpacing(spacing);
  }
  template <hermes::cuda::MemoryLocation LL> LevelSet2(LevelSet2<LL> &other) {
    grid_.copy(other.grid());
  }
  template <hermes::cuda::MemoryLocation LL> void copy(LevelSet2<LL> &other) {
    grid_.copy(other.grid());
  }
  /// \param resolution new resolution
  void setResolution(hermes::cuda::vec2u resolution) {
    grid_.resize(resolution);
  }
  /// \param spacing cell size
  void setSpacing(hermes::cuda::vec2f spacing) { grid_.setSpacing(spacing); }
  void setOrigin(hermes::cuda::point2f origin) { grid_.setOrigin(origin); }
  /// \return LevelSet2Accessor accessor for level set data
  LevelSet2Accessor accessor() { return LevelSet2Accessor(grid_.accessor()); }
  /// \return hermes::cuda::RegularGrid2<L, float>& grid data reference
  hermes::cuda::RegularGrid2<L, float> &grid() { return grid_; }
  /// Extracts isoline at 0 level
  /// \param vertices **[out]** output mesh
  /// \param isovalue **[in | default = 0]**
  void isoline(hermes::cuda::MemoryBlock1<L, float> &vertices,
               hermes::cuda::MemoryBlock1<L, unsigned int> &indices,
               float isovalue = 0.f) {
    hermes::cuda::MarchingSquares::extractIsoline(grid_, vertices, indices,
                                                  0.f);
  }

private:
  hermes::cuda::RegularGrid2<L, float> grid_;
};

/// Auxiliar class to access the level set properties
class LevelSet3Accessor {
public:
  LevelSet3Accessor(
      const hermes::cuda::RegularGrid3Accessor<float> &grid_accessor)
      : acc_(grid_accessor) {}
  __host__ __device__ hermes::cuda::vec3u resolution() const {
    return acc_.resolution();
  }
  __host__ __device__ hermes::cuda::vec3f spacing() const {
    return acc_.spacing();
  }
  __host__ __device__ float &operator()(int i, int j, int k) {
    return acc_(i, j, k);
  }
  __host__ __device__ const float &operator()(int i, int j, int k) const {
    return acc_(i, j, k);
  }
  __host__ __device__ hermes::cuda::vec3f gradient(int i, int j, int k) const {
    return gradientAt(acc_, i, j, k, 2);
  }
  __host__ __device__ hermes::cuda::point3f worldPosition(int i, int j, int k) {
    return acc_.worldPosition(i, j, k);
  }
  __host__ __device__ hermes::cuda::point3f
  gridPosition(const hermes::cuda::point3f &wp) {
    return acc_.gridPosition(wp);
  }
  __host__ __device__ bool isIndexStored(int i, int j, int k) {
    return acc_.isIndexStored(i, j, k);
  }

private:
  hermes::cuda::RegularGrid3Accessor<float> acc_;
};

/// Represent a field of distances from a isoline
/// \tparam L memory location
template <hermes::cuda::MemoryLocation L> class LevelSet3 {
public:
  LevelSet3() {}
  /// \brief Construct a new Level Set 3 object
  /// \param resolution regular grid resolution
  /// \param spacing point spacing
  /// \param origin offset
  LevelSet3(const hermes::cuda::vec3u &resolution,
            const hermes::cuda::vec3f &spacing,
            const hermes::cuda::point3f &origin = hermes::cuda::point3f(0.f)) {
    grid_.resize(resolution);
    grid_.setOrigin(origin);
    grid_.setSpacing(spacing);
  }
  template <hermes::cuda::MemoryLocation LL> LevelSet3(LevelSet3<LL> &other) {
    grid_.copy(other.grid());
  }
  template <hermes::cuda::MemoryLocation LL> void copy(LevelSet3<LL> &other) {
    grid_.copy(other.grid());
  }
  /// \param resolution new resolution
  void setResolution(hermes::cuda::vec3u resolution) {
    grid_.resize(resolution);
  }
  /// \param spacing cell size
  void setSpacing(hermes::cuda::vec3f spacing) { grid_.setSpacing(spacing); }
  void setOrigin(hermes::cuda::point3f origin) { grid_.setOrigin(origin); }
  /// \return LevelSet3Accessor accessor for level set data
  LevelSet3Accessor accessor() { return LevelSet3Accessor(grid_.accessor()); }
  /// \return hermes::cuda::RegularGrid3<L, float>& grid data reference
  hermes::cuda::RegularGrid3<L, float> &grid() { return grid_; }
  /// Extracts isoline at 0 level
  /// \param vertices **[out]** output mesh
  /// \param isovalue **[in | default = 0]**
  void isosurface(hermes::cuda::MemoryBlock1<L, float> &vertices,
                  hermes::cuda::MemoryBlock1<L, unsigned int> &indices,
                  float isovalue = 0.f,
                  hermes::cuda::MemoryBlock1<L, float> *normals = nullptr) {
    hermes::cuda::MarchingCubes::extractSurface(grid_, vertices, indices,
                                                isovalue, normals);
  }

private:
  hermes::cuda::RegularGrid3<L, float> grid_;
};

using LevelSet2H = LevelSet2<hermes::cuda::MemoryLocation::HOST>;
using LevelSet2D = LevelSet2<hermes::cuda::MemoryLocation::DEVICE>;
using LevelSet3H = LevelSet3<hermes::cuda::MemoryLocation::HOST>;
using LevelSet3D = LevelSet3<hermes::cuda::MemoryLocation::DEVICE>;

} // namespace cuda

} // namespace poseidon

#endif