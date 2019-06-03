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

// TODO: DEPRECATED
struct Grid2Info {
  Transform2<float> toField;
  Transform2<float> toWorld;
  vec2u resolution;
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
  vec2u resolution;
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
  __host__ __device__
  RegularGrid2Iterator(MemoryBlock2Accessor<T> &dataAccessor,
                       RegularGrid2Info &info, const vec2i &ij)
      : acc_(dataAccessor), info_(info), size_(info.resolution), i(ij.x),
        j(ij.y) {}
  __host__ __device__ vec2u size() const { return size_; }
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
  MemoryBlock2Accessor<T> acc_;
  RegularGrid2Info info_;
  int i = 0, j = 0;
  vec2u size_;
};

/// Accessor for arrays stored on the device.
/// \tparam T data type
template <typename T> class RegularGrid2Accessor {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid2Accessor(const RegularGrid2Info &info,
                       MemoryBlock2Accessor<T> data,
                       AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
                       T border = T(0))
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ vec2u resolution() const { return data_.size(); }
  __host__ __device__ vec2f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ T &operator()(int i, int j) {
    switch (address_mode_) {
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y) {
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
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y)
        return border_;
      break;
    case AddressMode::WRAP:
      break;
    case AddressMode::MIRROR:
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
  MemoryBlock2Accessor<T> data_;
  AddressMode address_mode_; //!< defines how out of bounds data is treated
  T border_;                 //!< border value
  T dummy_;                  //!< used as out of bounds reference variable
};

template <> class RegularGrid2Accessor<float> {
public:
  /// \param data raw pointer to device data
  /// \param addressMode **[default = AccessMode::NONE]** accessMode defines how
  /// outside of bounds is treated
  /// \param border * * [default = T()]** border
  RegularGrid2Accessor(const RegularGrid2Info &info,
                       MemoryBlock2Accessor<float> data,
                       AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
                       float border = 0.f)
      : info_(info), data_(data), address_mode_(addressMode), border_(border) {}
  __host__ __device__ vec2u resolution() const { return data_.size(); }
  __host__ __device__ vec2f spacing() const { return info_.spacing; }
  /// \param i size[0] index
  /// \param j size[1] index
  /// \return T& reference to data (a dummy variable is return in the case of an
  /// out of bounds index)
  __host__ __device__ float &operator()(int i, int j) {
    switch (address_mode_) {
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y) {
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
    case AddressMode::REPEAT:
      i = (i < 0) ? data_.size().x - 1 - i : i % data_.size().x;
      j = (j < 0) ? data_.size().y - 1 - j : j % data_.size().y;
      break;
    case AddressMode::CLAMP_TO_EDGE:
      i = fmaxf(0, fminf(i, data_.size().x - 1));
      j = fmaxf(0, fminf(j, data_.size().y - 1));
      break;
    case AddressMode::BORDER:
      if (i < 0 || i >= data_.size().x || j < 0 || j >= data_.size().y)
        return border_;
      break;
    case AddressMode::WRAP:
      break;
    case AddressMode::MIRROR:
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
  MemoryBlock2Accessor<float> data_;
  AddressMode address_mode_; //!< defines how out of bounds data is treated
  float border_;             //!< border value
  float dummy_;              //!< used as out of bounds reference variable
};                           // namespace cuda

/// Represents a regular grid that can be used in numeric calculations
template <MemoryLocation L, typename T> class RegularGrid2 {
public:
  template <MemoryLocation LL> RegularGrid2(RegularGrid2<LL, T> &other) {
    copy(other);
  }
  RegularGrid2(const vec2u &size = vec2u()) {
    info_.resolution = size;
    data_.resize(size);
    if (size.x * size.y != 0)
      data_.allocate();
  }
  /// Changes grid resolution
  /// \param res new resolution (in number of cells)
  void resize(const vec2u &res) {
    info_.resolution = res;
    data_.resize(res);
    data_.allocate();
  }
  vec2u resolution() const { return info_.resolution; }
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
  accessor(AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
           T border = T(0)) {
    return RegularGrid2Accessor<T>(info_, data_.accessor(), addressMode,
                                   border);
  }
  MemoryBlock2<L, T> &data() { return data_; }
  const MemoryBlock2<L, T> &data() const { return data_; }
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
  MemoryBlock2<L, T> data_;
};

using RegularGrid2Df = RegularGrid2<MemoryLocation::DEVICE, float>;
using RegularGrid2Duc = RegularGrid2<MemoryLocation::DEVICE, unsigned char>;
using RegularGrid2Di = RegularGrid2<MemoryLocation::DEVICE, int>;
using RegularGrid2Hf = RegularGrid2<MemoryLocation::HOST, float>;
using RegularGrid2Huc = RegularGrid2<MemoryLocation::HOST, unsigned char>;
using RegularGrid2Hi = RegularGrid2<MemoryLocation::HOST, int>;

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
  __host__ __device__ RegularGrid3Iterator<T> begin() {
    return RegularGrid3Iterator<T>(data_, info_, vec3i(0));
  }
  __host__ __device__ RegularGrid3Iterator<T> end() {
    return RegularGrid3Iterator<T>(data_, info_, vec3i(-1));
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
  __host__ __device__ RegularGrid3Iterator<float> begin() {
    return RegularGrid3Iterator<float>(data_, info_, vec3i(0));
  }
  __host__ __device__ RegularGrid3Iterator<float> end() {
    return RegularGrid3Iterator<float>(data_, info_, vec3i(-1));
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
  accessor(AddressMode addressMode = AddressMode::CLAMP_TO_EDGE,
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

template <MemoryLocation L, typename T> T min(RegularGrid2<L, T> &grid);

template <typename T> T min(RegularGrid2<MemoryLocation::DEVICE, T> &grid) {
  return min(grid.data());
}

template <MemoryLocation L, typename T> T max(RegularGrid2<L, T> &grid);

template <typename T> T max(RegularGrid2<MemoryLocation::DEVICE, T> &grid) {
  return max(grid.data());
}

// TODO: DEPRECATED
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