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

#ifndef POSEIDON_SIMULATION_CUDA_PARTICLE_SYSTEM_H
#define POSEIDON_SIMULATION_CUDA_PARTICLE_SYSTEM_H

#include <hermes/algorithms/cuda_marching_cubes.h>
#include <hermes/numeric/cuda_grid.h>
#include <hermes/numeric/cuda_numeric.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace poseidon {

namespace cuda {

class ParticleSystem2Iterator {
public:
  class Element {
  public:
    __host__ __device__ Element(hermes::cuda::point2f &p, size_t id, int &code,
                                char &active)
        : position(p), id_(id), code_(code), active_(active) {}
    __host__ __device__ size_t id() const { return id_; }
    __host__ __device__ int code() const { return code_; }
    __host__ __device__ void remove() { active_ = false; }
    __host__ __device__ bool isActive() const { return active_; }

    hermes::cuda::point2f &position;

  private:
    size_t id_;
    int &code_;
    char &active_;
  };
  __host__ __device__ ParticleSystem2Iterator(size_t count,
                                              hermes::cuda::point2f *p, int *c,
                                              char *a, int i = 0)
      : count_(count), positions_(p), codes_(c), active_(a), i(i) {}
  __host__ __device__ ParticleSystem2Iterator &operator++() {
    i++;
    if (i >= count_)
      i = -1;
    return *this;
  }
  __host__ __device__ Element operator*() {
    return Element(positions_[i], i, codes_[i], active_[i]);
  }
  __host__ __device__ bool operator!=(const ParticleSystem2Iterator &other) {
    // TODO: it is not enough
    return i != other.i;
  }

private:
  size_t count_;
  hermes::cuda::point2f *positions_;
  int *codes_;
  char *active_;
  int i = 0;
};

class ParticleSystem2Accessor {
public:
  /// \param to_grid **[in]**
  /// \param n_bits **[in]**
  /// \param count **[in]**
  /// \param p **[in]**
  /// \param c **[in]**
  /// \param a **[in]**
  /// \param on_device **[in]**
  ParticleSystem2Accessor(const hermes::cuda::Transform2f &to_grid,
                          size_t n_bits, size_t count, hermes::cuda::point2f *p,
                          int *c, char *a, bool on_device)
      : to_grid_(to_grid), n_bits_(n_bits), count_(count), positions_(p),
        codes_(c), active_(a), on_device_(on_device) {}
  /// \param wp **[in]**
  /// \return hermes::cuda::point2f
  __host__ __device__ hermes::cuda::point2f
  gridPosition(const hermes::cuda::point2f &wp) {
    return to_grid_(wp);
  }
  /// \return size_t
  __host__ __device__ size_t count() const { return count_; }
  /// \param index **[in]**
  /// \return const hermes::cuda::point2f&
  __host__ __device__ const hermes::cuda::point2f &
  position(size_t index) const {
    return positions_[index];
  }
  /// \param index **[in]**
  /// \return hermes::cuda::point2f&
  __host__ __device__ hermes::cuda::point2f &position(size_t index) {
    return positions_[index];
  }
  /// \param index **[in]**
  /// \return int
  __host__ __device__ int code(size_t index) const { return codes_[index]; }
  /// \param index **[in]**
  /// \return int&
  __host__ __device__ int &code(size_t index) { return codes_[index]; }
  /// \param index **[in]**
  __host__ __device__ void remove(size_t index) { active_[index] = 0; }
  /// \param index **[in]**
  /// \return true
  /// \return false
  __host__ __device__ bool isActive(size_t index) { return active_[index]; }
  /// \return size_t
  __host__ __device__ size_t nbits() const { return n_bits_; }
  /// \return int*
  __host__ __device__ int *codes() { return codes_; }
  /// \param wp **[in]**
  /// \return int
  __host__ __device__ int closestFrom(const hermes::cuda::point2f &wp) {
    return -1;
  }
  /// \param ids **[out]**
  /// \param size **[in]**
  /// \param i **[in]**
  /// \param j **[in]**
  __host__ __device__ void cellGridParticles(int *ids, int size, int i, int j) {
  }
  __host__ __device__ ParticleSystem2Iterator begin() {
    return ParticleSystem2Iterator(count_, positions_, codes_, active_, 0);
  }
  __host__ __device__ ParticleSystem2Iterator end() {
    return ParticleSystem2Iterator(count_, positions_, codes_, active_, -1);
  }

private:
  /// \param ids **[in]**
  /// \param size **[in]**
  /// \param zcode **[in]**
  /// \param treeLevel **[in]**
  /// \return int
  __device__ __host__ int searchParticles(int *ids, int size, int zcode,
                                          int treeLevel) {
    int *ptr = nullptr;
    if (on_device_)
      thrust::lower_bound(thrust::device, codes_, codes_ + count_, zcode);
    else
      thrust::lower_bound(thrust::host, codes_, codes_ + count_, zcode);
    int upperBound = zcode + (1 << ((n_bits_ - treeLevel) * 2));
    int count = 0;
    while (ptr != codes_ + count_ && *ptr >= zcode && *ptr < upperBound &&
           count < size) {
      ids[count++] = ptr - codes_;
      ptr++;
    }
    return count;
  }
  hermes::cuda::Transform2f to_grid_;
  size_t n_bits_;
  size_t count_;
  hermes::cuda::point2f *positions_;
  int *codes_;
  char *active_;
  bool on_device_;
};

template <hermes::cuda::MemoryLocation L> class ParticleSystem2 {};

template <> class ParticleSystem2<hermes::cuda::MemoryLocation::DEVICE> {
public:
  friend class ParticleSystem2<hermes::cuda::MemoryLocation::HOST>;
  ParticleSystem2();
  template <hermes::cuda::MemoryLocation L>
  ParticleSystem2(const ParticleSystem2<L> &other) {
    copy(other);
  }
  virtual ~ParticleSystem2();
  /// Adds a scalar field
  /// \return int property's id
  int addScalarProperty();
  /// Sets the value of the property in all particles
  /// \param propertyId
  /// \param value
  void setScalarProperty(int propertyId, float value);
  /// \param n **[in]**
  void resize(size_t n);
  /// 1 - Detects active particles
  /// 2 - Computes particle's morton code
  /// 3 - sort data based on morton code
  /// This method assumes data is on DEVICE side
  /// \param removedCount Count of previously removed particles externally
  void update(int removedCount = 0);
  /// \return ParticleSystem2Accessor
  ParticleSystem2Accessor accessor();
  /// \tparam L
  /// \param other **[in]**
  template <hermes::cuda::MemoryLocation L>
  void copy(const ParticleSystem2<L> &other);

private:
  // properties
  std::vector<thrust::device_vector<float>> d_scalar_properties_;
  // data
  thrust::device_vector<int> d_zcodes_;
  thrust::device_vector<char> d_active_;
  thrust::device_vector<hermes::cuda::point2f> d_positions_;
  hermes::cuda::Transform2f to_grid_; //!< map to underling grid
  size_t grid_resolution_ = 16;       //!< grid resolution for zcodes
  size_t active_count_ = 0;           //!< active particles count
  // size_t _nextId = 0;      //!< next particle id to be created
  size_t n_bits_ = 0;       //!< number of bits used by the maximum coordinate
  size_t max_depth_ = 0;    //!< maximum depth of search tree, bounded by nbits
  size_t max_z_code_ = 0;   //!< maximum allowed morton code
  bool need_update_ = true; ///< true if morton code must be recomputed
                            ///< and the data sorted again
};

template <> class ParticleSystem2<hermes::cuda::MemoryLocation::HOST> {
public:
  friend class ParticleSystem2<hermes::cuda::MemoryLocation::DEVICE>;
  ParticleSystem2();
  template <hermes::cuda::MemoryLocation L>
  ParticleSystem2(const ParticleSystem2<L> &other) {
    copy(other);
  }
  virtual ~ParticleSystem2();
  /// Adds a scalar field
  /// \return int property's id
  int addScalarProperty();
  /// Sets the value of the property in all particles
  /// \param propertyId
  /// \param value
  void setScalarProperty(int propertyId, float value);
  /// \param n **[in]**
  void resize(size_t n);
  /// 1 - Detects active particles
  /// 2 - Computes particle's morton code
  /// 3 - sort data based on morton code
  /// This method assumes data is on DEVICE side
  /// \param removedCount Count of previously removed particles externally
  void update(int removedCount = 0);
  /// \return ParticleSystem2Accessor
  ParticleSystem2Accessor accessor();
  /// \tparam L
  /// \param other **[in]**
  template <hermes::cuda::MemoryLocation L>
  void copy(const ParticleSystem2<L> &other);

private:
  // properties
  std::vector<thrust::host_vector<float>> h_scalar_properties_;
  // data
  thrust::host_vector<int> h_zcodes_;
  thrust::host_vector<char> h_active_;
  thrust::host_vector<hermes::cuda::point2f> h_positions_;
  hermes::cuda::Transform2f to_grid_; //!< map to underling grid
  size_t grid_resolution_ = 16;       //!< grid resolution for zcodes
  size_t active_count_ = 0;           //!< active particles count
  // size_t _nextId = 0;      //!< next particle id to be created
  size_t n_bits_ = 0;       //!< number of bits used by the maximum coordinate
  size_t max_depth_ = 0;    //!< maximum depth of search tree, bounded by nbits
  size_t max_z_code_ = 0;   //!< maximum allowed morton code
  bool need_update_ = true; ///< true if morton code must be recomputed
                            ///< and the data sorted again
};

using ParticleSystem2H = ParticleSystem2<hermes::cuda::MemoryLocation::HOST>;
using ParticleSystem2D = ParticleSystem2<hermes::cuda::MemoryLocation::DEVICE>;

} // namespace cuda

} // namespace poseidon

#endif