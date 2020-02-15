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

#ifndef POSEIDON_NUMERIC_CUDA_FD_H
#define POSEIDON_NUMERIC_CUDA_FD_H

#include <hermes/storage/cuda_memory_block.h>

namespace poseidon {

namespace cuda {

struct FDMatrix2Entry {
  float diag, x, y;
};

class FDMatrix2Accessor {
public:
  FDMatrix2Accessor(hermes::cuda::MemoryBlock2Accessor<FDMatrix2Entry> A,
                    hermes::cuda::MemoryBlock2Accessor<int> indices)
      : A_(A), indices_(indices) {}
  __host__ __device__ hermes::cuda::vec2u gridSize() const { return A_.size(); }
  __host__ __device__ size_t size() const { return A_.size().x * A_.size().y; }
  __host__ __device__ bool isIndexStored(int li, int lj, int ci, int cj) {
    if (!A_.isIndexValid(li, lj))
      return false;
    if (indices_(li, lj) < 0)
      return false;
    if ((int)fabsf(li - ci) + (int)fabsf(lj - cj) > 1)
      return false;
    if (!indices_.isIndexValid(ci, cj))
      return false;
    if (indices_(ci, cj) < 0)
      return false;
    return true;
  }
  __host__ __device__ float &operator()(int li, int lj, int ci, int cj) {
    dummy = 0.f;
    if (!indices_.isIndexValid(ci, cj))
      return dummy;
    if (indices_(ci, cj) < 0)
      return dummy;
    if (li == ci && lj == cj && A_.isIndexValid(li, lj) &&
        indices_(li, lj) >= 0)
      return A_(li, lj).diag;
    if (ci == li + 1 && cj == lj && A_.isIndexValid(li, lj) &&
        indices_(li, lj) >= 0)
      return A_(li, lj).x;
    if (ci + 1 == li && cj == lj && A_.isIndexValid(ci, cj) &&
        indices_(ci, cj) >= 0)
      return A_(ci, cj).x;
    if (ci == li && cj == lj + 1 && A_.isIndexValid(li, lj) &&
        indices_(li, lj) >= 0)
      return A_(li, lj).y;
    if (ci == li && cj + 1 == lj && A_.isIndexValid(ci, cj) &&
        indices_(ci, cj) >= 0)
      return A_(ci, cj).y;
    return dummy;
  }
  __host__ __device__ int elementIndex(int i, int j) const {
    if (!indices_.isIndexValid(i, j))
      return -1;
    return indices_(i, j);
  }

private:
  float dummy;
  hermes::cuda::MemoryBlock2Accessor<FDMatrix2Entry> A_;
  hermes::cuda::MemoryBlock2Accessor<int> indices_;
};

template <hermes::cuda::MemoryLocation L> class FDMatrix2 {
public:
  FDMatrix2() = default;
  FDMatrix2(const hermes::cuda::vec2u &size) { resize(size); }
  void resize(const hermes::cuda::vec2u &size) {
    data_.resize(size);
    data_.allocate();
    indices_.resize(size);
    indices_.allocate();
  }
  hermes::cuda::vec2u gridSize() const { return data_.size(); }
  size_t size() const { return data_.size().x * data_.size().y; }
  hermes::cuda::MemoryBlock2<L, FDMatrix2Entry> &data() { return data_; }
  hermes::cuda::MemoryBlock2Accessor<FDMatrix2Entry> dataAccessor() {
    return data_.accessor();
  }
  hermes::cuda::MemoryBlock2<L, int> &indexData() { return indices_; }
  hermes::cuda::MemoryBlock2Accessor<int> indexDataAccessor() {
    return indices_.accessor();
  }
  template <hermes::cuda::MemoryLocation LL> void copy(FDMatrix2<LL> &other) {
    hermes::cuda::memcpy(data_, other.data());
    hermes::cuda::memcpy(indices_, other.indexData());
  }
  FDMatrix2Accessor accessor() {
    return FDMatrix2Accessor(data_.accessor(), indices_.accessor());
  }

private:
  hermes::cuda::MemoryBlock2<L, FDMatrix2Entry> data_;
  hermes::cuda::MemoryBlock2<L, int> indices_;
};

struct FDMatrix3Entry {
  float diag, x, y, z;
};

class FDMatrix3Accessor {
public:
  FDMatrix3Accessor(hermes::cuda::MemoryBlock3Accessor<FDMatrix3Entry> A,
                    hermes::cuda::MemoryBlock3Accessor<int> indices)
      : A_(A), indices_(indices) {}
  __host__ __device__ hermes::cuda::vec3u gridSize() const { return A_.size(); }
  __host__ __device__ size_t size() const {
    return A_.size().x * A_.size().y * A_.size().z;
  }
  __host__ __device__ bool isIndexStored(int li, int lj, int lk, int ci, int cj,
                                         int ck) {
    if (!A_.isIndexValid(li, lj, lk))
      return false;
    if (indices_(li, lj, lk) < 0)
      return false;
    if ((int)fabsf(li - ci) + (int)fabsf(lj - cj) + (int)(fabsf(lk - ck)) > 1)
      return false;
    if (!indices_.isIndexValid(ci, cj, ck))
      return false;
    if (indices_(ci, cj, ck) < 0)
      return false;
    return true;
  }
  __host__ __device__ float &operator()(int li, int lj, int lk, int ci, int cj,
                                        int ck) {
    dummy = 0.f;
    if (!indices_.isIndexValid(ci, cj, ck))
      return dummy;
    if (indices_(ci, cj, ck) < 0)
      return dummy;
    if (li == ci && lj == cj && lk == ck && A_.isIndexValid(li, lj, lk) &&
        indices_(li, lj, lk) >= 0)
      return A_(li, lj, lk).diag;
    if (ci == li + 1 && cj == lj && ck == lk && A_.isIndexValid(li, lj, lk) &&
        indices_(li, lj, lk) >= 0)
      return A_(li, lj, lk).x;
    if (ci + 1 == li && cj == lj && ck == lk && A_.isIndexValid(ci, cj, ck) &&
        indices_(ci, cj, ck) >= 0)
      return A_(ci, cj, ck).x;
    if (ci == li && cj == lj + 1 && ck == lk && A_.isIndexValid(li, lj, lk) &&
        indices_(li, lj, lk) >= 0)
      return A_(li, lj, lk).y;
    if (ci == li && cj + 1 == lj && ck == lk && A_.isIndexValid(ci, cj, ck) &&
        indices_(ci, cj, ck) >= 0)
      return A_(ci, cj, ck).y;
    if (ci == li && cj == lj && ck == lk + 1 && A_.isIndexValid(li, lj, lk) &&
        indices_(li, lj, lk) >= 0)
      return A_(li, lj, lk).z;
    if (ci == li && cj == lj && ck + 1 == lk && A_.isIndexValid(ci, cj, ck) &&
        indices_(ci, cj, ck) >= 0)
      return A_(ci, cj, ck).z;
    return dummy;
  }
  __host__ __device__ int elementIndex(int i, int j, int k) const {
    if (!indices_.isIndexValid(i, j, k))
      return -1;
    return indices_(i, j, k);
  }

private:
  float dummy;
  hermes::cuda::MemoryBlock3Accessor<FDMatrix3Entry> A_;
  hermes::cuda::MemoryBlock3Accessor<int> indices_;
};

template <hermes::cuda::MemoryLocation L> class FDMatrix3 {
public:
  FDMatrix3() = default;
  FDMatrix3(const hermes::cuda::vec3u &size) { resize(size); }
  void resize(const hermes::cuda::vec3u &size) {
    data_.resize(size);
    data_.allocate();
    indices_.resize(size);
    indices_.allocate();
  }
  hermes::cuda::vec3u gridSize() const { return data_.size(); }
  size_t size() const {
    return data_.size().x * data_.size().y * data_.size().z;
  }
  hermes::cuda::MemoryBlock3<L, FDMatrix3Entry> &data() { return data_; }
  hermes::cuda::MemoryBlock3Accessor<FDMatrix3Entry> dataAccessor() {
    return data_.accessor();
  }
  hermes::cuda::MemoryBlock3<L, int> &indexData() { return indices_; }
  hermes::cuda::MemoryBlock3Accessor<int> indexDataAccessor() {
    return indices_.accessor();
  }
  template <hermes::cuda::MemoryLocation LL> void copy(FDMatrix3<LL> &other) {
    hermes::cuda::memcpy(data_, other.data());
    hermes::cuda::memcpy(indices_, other.indexData());
  }
  FDMatrix3Accessor accessor() {
    return FDMatrix3Accessor(data_.accessor(), indices_.accessor());
  }

private:
  hermes::cuda::MemoryBlock3<L, FDMatrix3Entry> data_;
  hermes::cuda::MemoryBlock3<L, int> indices_;
};

inline std::ostream &operator<<(std::ostream &os, FDMatrix2Accessor &A) {
  std::cerr << "FDMatrix2 (" << A.gridSize().x << " x " << A.gridSize().y
            << ") = (" << A.size() << " x " << A.size() << ")\n";
  for (int y = 0; y < A.gridSize().y; y++)
    for (int x = 0; x < A.gridSize().x; x++) {
      if (A.elementIndex(x, y) < 0)
        continue;
      for (int cy = 0; cy < A.gridSize().y; cy++)
        for (int cx = 0; cx < A.gridSize().x; cx++)
          if (A.elementIndex(cx, cy) >= 0)
            os << A(x, y, cx, cy) << " ";
      os << std::endl;
    }
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, FDMatrix2<hermes::cuda::MemoryLocation::HOST> &A) {
  std::cerr << "FDMatrix2 (" << A.gridSize().x << " x " << A.gridSize().y
            << ") = (" << A.size() << " x " << A.size() << ")\n";
  auto acc = A.dataAccessor();
  auto iAcc = A.indexDataAccessor();
  for (int y = 0; y < A.gridSize().y; y++)
    for (int x = 0; x < A.gridSize().x; x++)
      if (iAcc(x, y) >= 0) {
        os << "l[" << iAcc(x, y) << "]\t";
        if (iAcc.isIndexValid(x, y - 1) && iAcc(x, y - 1) >= 0)
          os << "(c" << iAcc(x, y - 1) << ", " << acc(x, y - 1).y << ") ";
        if (iAcc.isIndexValid(x - 1, y) && iAcc(x - 1, y) >= 0)
          os << "(c" << iAcc(x - 1, y) << ", " << acc(x - 1, y).x << ") ";
        os << "(c" << iAcc(x, y) << ", " << acc(x, y).diag << ") ";
        if (iAcc.isIndexValid(x + 1, y) && iAcc(x + 1, y) >= 0)
          os << "(c" << iAcc(x + 1, y) << ", " << acc(x, y).x << ") ";
        if (iAcc.isIndexValid(x, y + 1) && iAcc(x, y + 1) >= 0)
          os << "(c" << iAcc(x, y + 1) << ", " << acc(x, y).y << ") ";
        os << std::endl;
      }
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           FDMatrix2<hermes::cuda::MemoryLocation::DEVICE> &A) {
  FDMatrix2<hermes::cuda::MemoryLocation::HOST> host;
  host.resize(A.gridSize());
  host.copy(A);
  os << host << std::endl;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, FDMatrix3Accessor &A) {
  std::cerr << "FDMatrix3 (" << A.gridSize().x << " x " << A.gridSize().y
            << " x " << A.gridSize().z << ") = (" << A.size() << " x "
            << A.size() << ")\n";
  for (int z = 0; z < A.gridSize().z; z++)
    for (int y = 0; y < A.gridSize().y; y++)
      for (int x = 0; x < A.gridSize().x; x++) {
        if (A.elementIndex(x, y, z) < 0)
          continue;
        for (int cz = 0; cz < A.gridSize().z; cz++)
          for (int cy = 0; cy < A.gridSize().y; cy++)
            for (int cx = 0; cx < A.gridSize().x; cx++)
              if (A.elementIndex(cx, cy, cz) >= 0)
                os << A(x, y, z, cx, cy, cz) << " ";
        os << std::endl;
      }
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, FDMatrix3<hermes::cuda::MemoryLocation::HOST> &A) {
  std::cerr << "FDMatrix3 (" << A.gridSize().x << " x " << A.gridSize().y
            << " x " << A.gridSize().z << ") = (" << A.size() << " x "
            << A.size() << ")\n";
  auto acc = A.dataAccessor();
  auto iAcc = A.indexDataAccessor();
  for (int z = 0; z < A.gridSize().z; z++)
    for (int y = 0; y < A.gridSize().y; y++)
      for (int x = 0; x < A.gridSize().x; x++)
        if (iAcc(x, y, z) >= 0) {
          os << "l[" << iAcc(x, y, z) << "]\t";
          if (iAcc.isIndexValid(x, y, z - 1) && iAcc(x, y, z - 1) >= 0)
            os << "(c" << iAcc(x, y, z - 1) << ", " << acc(x, y, z - 1).z
               << ") ";
          if (iAcc.isIndexValid(x, y - 1, z) && iAcc(x, y - 1, z) >= 0)
            os << "(c" << iAcc(x, y - 1, z) << ", " << acc(x, y - 1, z).y
               << ") ";
          if (iAcc.isIndexValid(x - 1, y, z) && iAcc(x - 1, y, z) >= 0)
            os << "(c" << iAcc(x - 1, y, z) << ", " << acc(x - 1, y, z).x
               << ") ";
          os << "(c" << iAcc(x, y, z) << ", " << acc(x, y, z).diag << ") ";
          if (iAcc.isIndexValid(x + 1, y, z) && iAcc(x + 1, y, z) >= 0)
            os << "(c" << iAcc(x + 1, y, z) << ", " << acc(x, y, z).x << ") ";
          if (iAcc.isIndexValid(x, y + 1, z) && iAcc(x, y + 1, z) >= 0)
            os << "(c" << iAcc(x, y + 1, z) << ", " << acc(x, y, z).y << ") ";
          if (iAcc.isIndexValid(x, y, z + 1) && iAcc(x, y, z + 1) >= 0)
            os << "(c" << iAcc(x, y, z + 1) << ", " << acc(x, y, z).z << ") ";
          os << std::endl;
        }
  return os;
}

inline std::ostream &
operator<<(std::ostream &os,
           FDMatrix3<hermes::cuda::MemoryLocation::DEVICE> &A) {
  FDMatrix3<hermes::cuda::MemoryLocation::HOST> host;
  host.resize(A.gridSize());
  host.copy(A);
  os << host << std::endl;
  return os;
}

template <hermes::cuda::MemoryLocation L, typename T>
void mul(FDMatrix2<L> &A, hermes::cuda::CuMemoryBlock1<T> &x,
         hermes::cuda::CuMemoryBlock1<T> &b);
template <hermes::cuda::MemoryLocation L, typename T>
void mul(FDMatrix3<L> &A, hermes::cuda::CuMemoryBlock1<T> &x,
         hermes::cuda::CuMemoryBlock1<T> &b);

using FDMatrix2D = FDMatrix2<hermes::cuda::MemoryLocation::DEVICE>;
using FDMatrix2H = FDMatrix2<hermes::cuda::MemoryLocation::HOST>;
using FDMatrix3D = FDMatrix3<hermes::cuda::MemoryLocation::DEVICE>;
using FDMatrix3H = FDMatrix3<hermes::cuda::MemoryLocation::HOST>;

} // namespace cuda

} // namespace poseidon

#endif