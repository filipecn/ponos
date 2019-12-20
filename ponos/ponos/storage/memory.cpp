/*
 * Copyright (c) 2019 FilipeCN
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

#include <ponos/storage/memory.h>

namespace ponos {

void *allocAligned(size_t size) {
#ifdef _WIN32
  return _aligned_malloc(PONOS_CACHE_L1_LINE_SIZE, size);
#elif defined(__APPLE__)
  void *ptr;
  if (posix_memalign(&ptr, PONOS_CACHE_L1_LINE_SIZE, size) != 0)
    ptr = nullptr;
  return ptr;
#elif defined(__linux)
  return memalign(PONOS_CACHE_L1_LINE_SIZE, size);
#endif
  return nullptr;
}

void freeAligned(void *ptr) {
  if (!ptr)
    return;
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

MemoryArena::MemoryArena(size_t bs) : blockSize(bs) {}

MemoryArena::~MemoryArena() {
  freeAligned(currentBlock);
  for (auto &block : usedBlocks)
    freeAligned((void *)block.second);
  for (auto &block : availableBlocks)
    freeAligned(block.second);
}

void *MemoryArena::alloc(size_t sz) {
  // round up sz to minimum machine alignment
  sz = ((sz + 15) & (~15));
  if (currentBlockPos + sz > currentAllocSize) {
    // add current block to usedBlocks list
    if (currentBlock) {
      usedBlocks.push_back(std::make_pair(currentAllocSize, currentBlock));
      currentBlock = nullptr;
    }
    // get new block of memory for MemoryArena
    // try to get memory block from avaliableBlocks
    for (auto iter = availableBlocks.begin(); iter != availableBlocks.end();
         ++iter) {
      if (iter->first >= sz) {
        currentAllocSize = iter->first;
        currentBlock = iter->second;
        availableBlocks.erase(iter);
        break;
      }
    }
    if (!currentBlock) {
      currentAllocSize = std::max(sz, blockSize);
      currentBlock = allocAligned<uint8_t>(currentAllocSize);
    }
    currentBlockPos = 0;
  }
  void *ret = currentBlock + currentBlockPos;
  currentBlockPos += sz;
  return ret;
}

void MemoryArena::freeAll() {
  currentBlockPos = 0;
  availableBlocks.splice(availableBlocks.begin(), usedBlocks);
}

} // namespace ponos