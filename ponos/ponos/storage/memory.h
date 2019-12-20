/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
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

#ifndef PONOS_MEMORY_H
#define PONOS_MEMORY_H

#include <ponos/common/defs.h>

#include <algorithm>
#include <cstdlib>
#ifdef __linux__
#include <malloc.h>
#endif
#include <list>
#include <vector>

namespace ponos {

template <typename T> class IndexPointerInterface {
public:
  virtual T *get(size_t i) = 0;
};

template <typename T> class IndexPointer {
public:
  IndexPointer() : array(nullptr), i(0) {}
  IndexPointer(size_t _i, IndexPointerInterface<T> *a) : array(a), i(_i) {}
  T *operator->() { return array->get(i); }
  size_t getIndex() { return i; }

private:
  IndexPointerInterface<T> *array;
  size_t i;
};

#ifndef PONOS_CACHE_L1_LINE_SIZE
#define PONOS_CACHE_L1_LINE_SIZE 64
#endif

/// Convenient wrapper for calling alloca
#define ALLOCA(TYPE, COUNT) (TYPE *)alloca((COUNT) * sizeof(TYPE))

/// Allocates cache-aligned memory blocks of **size** bytes.
/// \param size bytes
/// \returns pointer to allocated region.
void *allocAligned(size_t size);
/// Frees aligned memory
/// \param ptr raw pointer
void freeAligned(void *ptr);
/// \brief Allocates cache-aligned **count** objects.
/// \param count number of objects.
/// \returns pointer to allocated region.
template <typename T> T *allocAligned(u32 count) {
  return static_cast<T *>(allocAligned(count * sizeof(T)));
}

/// MemoryArena allocates a large contiguous region of memory and manages
/// objects of different size in this region. Does not support freeing of
/// individual blocks.
class MemoryArena {
public:
  /// \param bs **[default = 256kB]** size of chunks allocated by MemoryArena
  MemoryArena(size_t bs = 262144);
  virtual ~MemoryArena();
  /// Allocates **sz** bytes
  /// \param sz number of bytes to be allocated
  /// \return void*
  void *alloc(size_t sz);
  /// Allocates memory for **count** objects of type **T**
  /// \tparam T object type
  /// \param count number of objects
  /// \param runConstructor true to run constructor of objects
  /// \return T* pointer to the first object in array
  template <typename T> T *alloc(size_t count = 1, bool runConstructor = true) {
    T *ret = static_cast<T *>(alloc(count * sizeof(T)));
    if (runConstructor)
      for (u32 i = 0; i < count; i++)
        new (&ret[i]) T();
    return ret;
  }
  /// frees all allocated memory
  void freeAll();

private:
  const size_t blockSize;      //!< size of chunks allocated by MemoryArena
  size_t currentBlockPos = 0;  //!< offset of the first free location
  size_t currentAllocSize = 0; //!< total size of the current block allocation
  uint8_t *currentBlock = nullptr; //!< pointer to the current block of memory
  std::list<std::pair<size_t, uint8_t *>> usedBlocks; //!< fully used blocks
  std::list<std::pair<size_t, uint8_t *>>
      availableBlocks; //!< allocated but not used blocks
};

/** \brief blocked 2D arrays
 *
 * Implements a generic 2D array of values. Rather than storing values so that
 *entire
 * rows are contiguous in memory, the data is ordered in memory using a
 *_blocked_
 * memory layout. The array is subdivided into square blocks of BLOCK_SIZE
 * (in power of 2).
 *
 * **logBlockSize** is specified the logarithm (base 2) of the block size.
 */
template <typename T, int logBlockSize = 2> class BlockedArray {
public:
  BlockedArray(u32 nu, u32 nv, const T *d = nullptr) {
    uRes = nu;
    vRes = nv;
    uBlocks = roundUp(uRes) >> logBlockSize;
    u32 nAlloc = roundUp(uRes) * roundUp(vRes);
    data = allocAligned<T>(nAlloc);
    for (u32 i = 0; i < nAlloc; i++)
      new (&data[i]) T();
    if (d != nullptr)
      for (u32 v = 0; v < vRes; ++v)
        for (u32 u = 0; u < uRes; ++u)
          (*this)(u, v) = d[v * uRes + u];
  }
  /** \brief get
   * \returns block size (1 << **logBlockSize**)
   */
  u32 blockSize() const { return 1 << logBlockSize; }
  /** \brief rounding
   * \param x **[in]** quantity to be rounded
   * Rounds both dimensions up to the a multiple of the block size.
   * \returns rounded value
   */
  u32 roundUp(u32 x) const {
    return (x + blockSize() - 1) & !(blockSize() - 1);
  }
  /** \brief get
   * \returns u dimension size
   */
  u32 uSize() { return uRes; }
  /** \brief get
   * \returns v dimension size
   */
  u32 vSize() { return vRes; }

  u32 blockNumber(u32 a) const { return a >> logBlockSize; }

  u32 blockOffset(u32 a) const { return (a & (blockSize() - 1)); }

  T &operator()(u32 u, u32 v) {
    u32 bu = blockNumber(u), bv = blockNumber(v);
    u32 ou = blockOffset(u), ov = blockOffset(v);
    u32 offset = blockSize() * blockSize() * (uBlocks * bv + bu);
    offset += blockSize() * ov + ou;
    return data[offset];
  }

private:
  T *data;
  u32 uRes, vRes, uBlocks;
};

} // namespace ponos

#endif
