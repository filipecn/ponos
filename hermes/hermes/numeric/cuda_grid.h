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

namespace hermes {

namespace cuda {

struct Grid2Info {
  Transform2<float> toField;
  Transform2<float> toWorld;
  vec2u resolution;
  point2f origin;
  float dx;
};

struct Grid3Info {
  Transform<float> toField;
  Transform<float> toWorld;
  vec3u resolution;
  point3f origin;
  float dx;
};

struct RegularGrid3Info {
  Transform<float> toGrid;
  Transform<float> toWorld;
  vec3u resolution;
  point3f origin;
  vec3f spacing;
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
  RegularGrid3Accessor(const RegularGrid3Info &info,
                       MemoryBlock3Accessor<T> data,
                       AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
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
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z) {
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
    return data_(i, j, k);
  }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \param k size[2] index
  /// \return const T& reference to data (a dummy variable is return in the case
  /// of an out of bounds index)
  __host__ __device__ const T &operator()(int i, int j, int k) const {
    switch (address_mode_) {
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z)
        return border_;
      break;
    case AddressMode::WRAP:
      break;
    case AddressMode::MIRROR:
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

private:
  RegularGrid3Info info_;
  MemoryBlock3Accessor<T> data_;
  AddressMode address_mode_; //!< defines how out of bounds data is treated
  T border_;                 //!< border value
  T dummy_;                  //!< used as out of bounds reference variable
};

template <> class RegularGrid3Accessor<float> {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid3Accessor(const RegularGrid3Info &info,
                       MemoryBlock3Accessor<float> data,
                       AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
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
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z) {
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
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      k = (k < 0) ? data_.size().z - 1 - k : k % data_.size().z;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      k = fmaxf(0, fminf(k, data_.size().z - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y ||
          k < 0 || k >= data_.size().z)
        return border_;
      break;
    case AddressMode::WRAP:
      break;
    case AddressMode::MIRROR:
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

private:
  RegularGrid3Info info_;
  MemoryBlock3Accessor<float> data_;
  AddressMode address_mode_; //!< defines how out of bounds data is treated
  float border_;             //!< border value
  float dummy_;              //!< used as out of bounds reference variable
};

/// Represents a regular grid that can be used in numeric calculations
template <MemoryLocation L, typename T> class RegularGrid3 {
public:
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
  accessor(AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
           T border = T(0)) {
    return RegularGrid3Accessor<T>(info_, data_.accessor(), addressMode,
                                   border);
  }
  MemoryBlock3<L, T> &data() { return data_; }
  const MemoryBlock3<L, T> &data() const { return data_; }
  const RegularGrid3Info &info() const { return info_; }

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
                        bool overwrite) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (acc.isIndexStored(x, y, z)) {
    auto wp = acc.worldPosition(x, y, z);
    if (region.contains(wp)) {
      if (overwrite)
        acc(x, y, z) = value;
      else
        acc(x, y, z) += value;
    }
  }
}

template <typename T>
void fill3(RegularGrid3<MemoryLocation::DEVICE, T> &grid, const bbox3f &region,
           T value, bool overwrite = false) {
  ThreadArrayDistributionInfo td(grid.resolution());
  __fill3<<<td.gridSize, td.blockSize>>>(grid.accessor(), region, value,
                                         overwrite);
}

/// Represents a texture field with position offset and scale
template <typename T> class GridTexture2 {
public:
  GridTexture2() = default;
  /// \param resolution in number of cells
  /// \param origin (0,0) coordinate position
  /// \param dx cell size
  GridTexture2(vec2u resolution, point2f origin, float dx)
      : origin(origin), dx(dx) {
    texGrid.resize(resolution);
    updateTransform();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(vec2u res) { texGrid.resize(res); }
  vec2u resolution() const { return texGrid.resolution(); }
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
            vec2u(texGrid.texture().width(), texGrid.texture().height()),
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