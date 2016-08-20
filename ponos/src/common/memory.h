#ifndef PONOS_MEMORY_H
#define PONOS_MEMORY_H

#include "common/defs.h"

#include <cstdlib>
#include <malloc.h>
#include <vector>

namespace ponos {

#ifndef CACHE_L1_LINE_SIZE
#define CACHE_L1_LINE_SIZE 64
#endif

#define ALLOCA(TYPE, COUNT) (TYPE *)alloca((COUNT) * sizeof(TYPE))

	/*allocAligned allocate cache-aligned memory blocks of <size> bytes.
	 *@size bytes
	 */
	inline void* allocAligned(uint32 size) {
		return memalign(CACHE_L1_LINE_SIZE, size);
	}

	/*allocAligned allocate cache-aligned <count> objects.
	 *@count number of objects.
	 */
	template<typename T> T* allocAligned(uint32 count) {
		return static_cast<T*>(allocAligned(count * sizeof(T)));
	}

	inline void freeAligned(void *ptr) {
		free(ptr);
	}

	/*MemoryArena allocates a large contiguous region of memory and manages
	 * objects of different size in this region. Does not support freeing of
	 * individual blocks.
	 */
	class MemoryArena {
		public:
			/*@bs amount of bytes the region to be allocated (<blockSize>).
			 */
			MemoryArena(uint32 bs = 32768) {
				blockSize = bs;
				curBlockPos = 0;
				currentBlock = allocAligned<char>(blockSize);
			}
			virtual ~MemoryArena() {}

			/*alloc allocates <sz> bytes.
			 * @sz block size to be allocated.
			 * @return pointer to the allocated area. If there is not enough space
			 * a new block of <blockSize> is allocated.
			 */
			void * alloc(uint32 sz) {
				// round up to minimum machine alignment
				sz = ((sz + 15) & (~15));
				if(curBlockPos + sz > blockSize) {
					// get new block of memory for MemoryArena
					usedBlocks.emplace_back(currentBlock);
					if(availableBlocks.size() && sz <= blockSize) {
						currentBlock = availableBlocks.back();
						availableBlocks.pop_back();
					}
					else currentBlock = allocAligned<char>(std::max(sz, blockSize));
					curBlockPos = 0;
				}
				void *ret = currentBlock + curBlockPos;
				curBlockPos += sz;
				return ret;
			}

			/* alloc allocates space for <count> objects.
			 * @return pointer of the first object in the list.
			 */
			template<typename T> T * alloc(uint32 count = 1) {
				T *ret = static_cast<T *>(alloc(count * sizeof(T)));
				for(uint32 i = 0; i < count; i++)
					new (&ret[i]) T();
				return ret;
			}

			void freeAll() {
				curBlockPos = 0;
				while(usedBlocks.size()) {
					availableBlocks.emplace_back(usedBlocks.back());
					usedBlocks.pop_back();
				}
			}

		private:
			uint32 curBlockPos, blockSize;
			char *currentBlock;
			std::vector<char *> usedBlocks, availableBlocks;

	};

} // ponos namespace

#endif

