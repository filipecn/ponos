#ifndef PONOS_MEMORY_H
#define PONOS_MEMORY_H

#include "common/defs.h"

#include <algorithm>
#include <cstdlib>
#include <malloc.h>
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

#ifndef CACHE_L1_LINE_SIZE
#define CACHE_L1_LINE_SIZE 64
#endif

#define ALLOCA(TYPE, COUNT) (TYPE *)alloca((COUNT) * sizeof(TYPE))

/** \brief allocAligned
 *
 * Allocates cache-aligned memory blocks of **size** bytes.
 * \param size bytes
 */
inline void *allocAligned(uint32 size) {
#ifdef _WIN32
  return _aligned_malloc(CACHE_L1_LINE_SIZE, size);
#else
  return memalign(CACHE_L1_LINE_SIZE, size);
#endif
}

/** \brief allocAligned
 *
 * Allocates cache-aligned **count** objects.
 * \param count number of objects.
 */
template <typename T> T *allocAligned(uint32 count) {
  return static_cast<T *>(allocAligned(count * sizeof(T)));
}

inline void freeAligned(void *ptr) { free(ptr); }

/** \brief arena-based allocation
 *
 * MemoryArena allocates a large contiguous region of memory and manages
 * objects of different size in this region. Does not support freeing of
 * individual blocks.
 */
class MemoryArena {
public:
  /** \brief Constructor.
   * \param bs **[in]** amount of bytes per region to be allocated
   * (**blockSize**).
   */
  MemoryArena(uint32 bs = 32768) {
    blockSize = bs;
    curBlockPos = 0;
    currentBlock = allocAligned<char>(blockSize);
  }
  virtual ~MemoryArena() {}

  /** \brief allocation
   * \param sz **[in]** block size to be allocated.
   * \returns pointer to the allocated area. If there is not enough space.. //
   * TODO
   * Allocates **sz** bytes.
   */
  void *alloc(uint32 sz) {
    // round up to minimum machine alignment
    sz = ((sz + 15) & (~15));
    if (curBlockPos + sz > blockSize) {
      // get new block of memory for MemoryArena
      usedBlocks.emplace_back(currentBlock);
      if (availableBlocks.size() && sz <= blockSize) {
        currentBlock = availableBlocks.back();
        availableBlocks.pop_back();
      } else
        currentBlock = allocAligned<char>(std::max(sz, blockSize));
      curBlockPos = 0;
    }
    void *ret = currentBlock + curBlockPos;
    curBlockPos += sz;
    return ret;
  }

  /** \brief allocation
   * \param count **[in]** number of objects
   * alloc allocates space for **count** objects.
   * \returns pointer of the first object in the list.
   */
  template <typename T> T *alloc(uint32 count = 1) {
    T *ret = static_cast<T *>(alloc(count * sizeof(T)));
    for (uint32 i = 0; i < count; i++)
      new (&ret[i]) T();
    return ret;
  }

  /** free
   *
   * frees all allocated memory
   */
  void freeAll() {
    curBlockPos = 0;
    while (usedBlocks.size()) {
      availableBlocks.emplace_back(usedBlocks.back());
      usedBlocks.pop_back();
    }
  }

private:
  uint32 curBlockPos, blockSize;
  char *currentBlock;
  std::vector<char *> usedBlocks, availableBlocks;
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
  BlockedArray(uint32 nu, uint32 nv, const T *d = nullptr) {
    uRes = nu;
    vRes = nv;
    uBlocks = roundUp(uRes) >> logBlockSize;
    uint32 nAlloc = roundUp(uRes) * roundUp(vRes);
    data = allocAligned<T>(nAlloc);
    for (uint32 i = 0; i < nAlloc; i++)
      new (&data[i]) T();
    if (d != nullptr)
      for (uint32 v = 0; v < vRes; ++v)
        for (uint32 u = 0; u < uRes; ++u)
          (*this)(u, v) = d[v * uRes + u];
  }
  /** \brief get
   * \returns block size (1 << **logBlockSize**)
   */
  uint32 blockSize() const { return 1 << logBlockSize; }
  /** \brief rounding
   * \param x **[in]** quantity to be rounded
   * Rounds both dimensions up to the a multiple of the block size.
   * \returns rounded value
   */
  uint32 roundUp(uint32 x) const {
    return (x + blockSize() - 1) & !(blockSize() - 1);
  }
  /** \brief get
   * \returns u dimension size
   */
  uint32 uSize() { return uRes; }
  /** \brief get
   * \returns v dimension size
   */
  uint32 vSize() { return vRes; }

  uint32 blockNumber(uint32 a) const { return a >> logBlockSize; }

  uint32 blockOffset(uint32 a) const { return (a & (blockSize() - 1)); }

  T &operator()(uint32 u, uint32 v) {
    uint32 bu = blockNumber(u), bv = blockNumber(v);
    uint32 ou = blockOffset(u), ov = blockOffset(v);
    uint32 offset = blockSize() * blockSize() * (uBlocks * bv + bu);
    offset += blockSize() * ov + ou;
    return data[offset];
  }

private:
  T *data;
  uint32 uRes, vRes, uBlocks;
};

} // ponos namespace

#endif
