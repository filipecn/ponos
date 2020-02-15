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

#include <poseidon/simulation/cuda_particle_system2.h>
#include <thrust/gather.h>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

__global__ void __computeMortonCode(ParticleSystem2Accessor ps) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < ps.count()) {
    point2f p = ps.gridPosition(ps.position(index));
    ps.code(index) = mortonCode(p);
  }
}

ParticleSystem2<MemoryLocation::DEVICE>::ParticleSystem2() : active_count_(0) {
  max_z_code_ = mortonCode(point2f(grid_resolution_));
  int n = grid_resolution_ - 1;
  for (n_bits_ = 0; n; n >>= 1)
    n_bits_++;
  max_depth_ = n_bits_;
  vec2f scalePoint, regionSize(1.);
  for (int i = 0; i < 2; i++)
    scalePoint[i] = 16 / regionSize[i];
  to_grid_ = scale(scalePoint.x, scalePoint.y);
}

ParticleSystem2<MemoryLocation::DEVICE>::~ParticleSystem2() {}

int ParticleSystem2<MemoryLocation::DEVICE>::addScalarProperty() {
  int idx = d_scalar_properties_.size();
  d_scalar_properties_.push_back(thrust::device_vector<float>());
  return idx;
}

void ParticleSystem2<MemoryLocation::DEVICE>::setScalarProperty(int propertyId,
                                                                float value) {
  //   setPropertyValue<float>(
  //       thrust::raw_pointer_cast(d_scalar_properties_[propertyId].data()),
  //       active_count_, value);
}

// void ParticleSystem2<MemoryLocation::DEVICE>::remove(size_t i) {
//   h_active[i] = 0;
//   active_count_--;
//   need_update_ = true;
// }

void ParticleSystem2<MemoryLocation::DEVICE>::resize(size_t n) {
  if (n >= d_zcodes_.size()) {
    d_positions_.resize(n);
    d_active_.resize(n, 1);
    d_zcodes_.resize(n);
    for (size_t i = 0; i < d_scalar_properties_.size(); i++)
      d_scalar_properties_[i].resize(n, 0.f);
  }
  active_count_ = n;
  need_update_ = true;
}

void ParticleSystem2<MemoryLocation::DEVICE>::update(int removedCount) {
  if (!active_count_)
    return;
  {
    thrust::counting_iterator<int> iter(0);
    thrust::device_vector<int> indices(d_active_.size());
    thrust::copy(iter, iter + indices.size(), indices.begin());
    // --- First, sort the keys and indices by the keys
    thrust::sort_by_key(d_active_.begin(), d_active_.end(), indices.begin(),
                        thrust::greater<char>());
    // Now reorder the ID arrays using the sorted indices
    thrust::device_vector<point2f> tmp(d_positions_);
    thrust::gather(indices.begin(), indices.end(), tmp.begin(),
                   d_positions_.begin());
    // reorder particle properties
    for (size_t i = 0; i < d_scalar_properties_.size(); i++) {
      thrust::device_vector<float> aux(d_scalar_properties_[i]);
      thrust::gather(indices.begin(), indices.end(), aux.begin(),
                     d_scalar_properties_[i].begin());
    }
  }
  active_count_ -= removedCount;
  { // compute morton code
    hermes::ThreadArrayDistributionInfo td(active_count_);
    __computeMortonCode<<<td.gridSize, td.blockSize>>>(accessor());
  }
  { // sort based on morton code
    thrust::counting_iterator<int> iter(0);
    thrust::device_vector<int> indices(active_count_);
    thrust::copy(iter, iter + indices.size(), indices.begin());
    // First, sort the keys and indices by the keys
    thrust::sort_by_key(d_zcodes_.begin(), d_zcodes_.end(), indices.begin());
    // Now reorder the ID arrays using the sorted indices
    thrust::device_vector<point2f> tmp(d_positions_);
    thrust::gather(indices.begin(), indices.end(), tmp.begin(),
                   d_positions_.begin());
    // reorder particle properties
    for (size_t i = 0; i < d_scalar_properties_.size(); i++) {
      thrust::device_vector<float> aux(d_scalar_properties_[i]);
      thrust::gather(indices.begin(), indices.end(), aux.begin(),
                     d_scalar_properties_[i].begin());
    }
  }
}

ParticleSystem2Accessor ParticleSystem2<MemoryLocation::DEVICE>::accessor() {
  return ParticleSystem2Accessor(to_grid_, n_bits_, active_count_,
                                 thrust::raw_pointer_cast(d_positions_.data()),
                                 thrust::raw_pointer_cast(d_zcodes_.data()),
                                 thrust::raw_pointer_cast(d_active_.data()),
                                 true);
}

template <>
void ParticleSystem2<MemoryLocation::DEVICE>::copy(
    const ParticleSystem2<MemoryLocation::DEVICE> &other) {
  d_zcodes_ = other.d_zcodes_;
}

template <>
void ParticleSystem2<MemoryLocation::DEVICE>::copy(
    const ParticleSystem2<MemoryLocation::HOST> &other) {}

///////////////////////////////////////HOST/////////////////////////////////////

ParticleSystem2<MemoryLocation::HOST>::ParticleSystem2() : active_count_(0) {
  max_z_code_ = mortonCode(point2f(grid_resolution_));
  int n = grid_resolution_ - 1;
  for (n_bits_ = 0; n; n >>= 1)
    n_bits_++;
  max_depth_ = n_bits_;
  vec2f scalePoint, regionSize(1.);
  for (int i = 0; i < 2; i++)
    scalePoint[i] = 16 / regionSize[i];
  to_grid_ = scale(scalePoint.x, scalePoint.y);
}

ParticleSystem2<MemoryLocation::HOST>::~ParticleSystem2() {}

int ParticleSystem2<MemoryLocation::HOST>::addScalarProperty() {
  int idx = h_scalar_properties_.size();
  h_scalar_properties_.push_back(thrust::host_vector<float>());
  return idx;
}

void ParticleSystem2<MemoryLocation::HOST>::setScalarProperty(int propertyId,
                                                              float value) {
  //   setPropertyValue<float>(
  //       thrust::raw_pointer_cast(h_scalar_properties_[propertyId].data()),
  //       active_count_, value);
}

// void ParticleSystem2<MemoryLocation::HOST>::remove(size_t i) {
//   h_active[i] = 0;
//   active_count_--;
//   need_update_ = true;
// }

void ParticleSystem2<MemoryLocation::HOST>::resize(size_t n) {
  if (n >= h_zcodes_.size()) {
    h_positions_.resize(n);
    h_active_.resize(n, 1);
    h_zcodes_.resize(n);
    for (size_t i = 0; i < h_scalar_properties_.size(); i++)
      h_scalar_properties_[i].resize(n, 0.f);
  }
  active_count_ = n;
  need_update_ = true;
}

void ParticleSystem2<MemoryLocation::HOST>::update(int removedCount) {
  if (!active_count_)
    return;
  {
    thrust::counting_iterator<int> iter(0);
    thrust::host_vector<int> indices(h_active_.size());
    thrust::copy(iter, iter + indices.size(), indices.begin());
    // --- First, sort the keys and indices by the keys
    thrust::sort_by_key(h_active_.begin(), h_active_.end(), indices.begin(),
                        thrust::greater<char>());
    // Now reorder the ID arrays using the sorted indices
    thrust::host_vector<point2f> tmp(h_positions_);
    thrust::gather(indices.begin(), indices.end(), tmp.begin(),
                   h_positions_.begin());
    // reorder particle properties
    for (size_t i = 0; i < h_scalar_properties_.size(); i++) {
      thrust::host_vector<float> aux(h_scalar_properties_[i]);
      thrust::gather(indices.begin(), indices.end(), aux.begin(),
                     h_scalar_properties_[i].begin());
    }
  }
  active_count_ -= removedCount;
  { // compute morton code
    hermes::ThreadArrayDistributionInfo td(active_count_);
    __computeMortonCode<<<td.gridSize, td.blockSize>>>(accessor());
  }
  { // sort based on morton code
    thrust::counting_iterator<int> iter(0);
    thrust::host_vector<int> indices(active_count_);
    thrust::copy(iter, iter + indices.size(), indices.begin());
    // First, sort the keys and indices by the keys
    thrust::sort_by_key(h_zcodes_.begin(), h_zcodes_.end(), indices.begin());
    // Now reorder the ID arrays using the sorted indices
    thrust::host_vector<point2f> tmp(h_positions_);
    thrust::gather(indices.begin(), indices.end(), tmp.begin(),
                   h_positions_.begin());
    // reorder particle properties
    for (size_t i = 0; i < h_scalar_properties_.size(); i++) {
      thrust::host_vector<float> aux(h_scalar_properties_[i]);
      thrust::gather(indices.begin(), indices.end(), aux.begin(),
                     h_scalar_properties_[i].begin());
    }
  }
}

ParticleSystem2Accessor ParticleSystem2<MemoryLocation::HOST>::accessor() {
  return ParticleSystem2Accessor(to_grid_, n_bits_, active_count_,
                                 thrust::raw_pointer_cast(h_positions_.data()),
                                 thrust::raw_pointer_cast(h_zcodes_.data()),
                                 thrust::raw_pointer_cast(h_active_.data()),
                                 false);
}

template <>
void ParticleSystem2<MemoryLocation::HOST>::copy(
    const ParticleSystem2<MemoryLocation::DEVICE> &other) {
  h_zcodes_ = other.d_zcodes_;
}

template <>
void ParticleSystem2<MemoryLocation::HOST>::copy(
    const ParticleSystem2<MemoryLocation::HOST> &other) {}

} // namespace cuda

} // namespace poseidon