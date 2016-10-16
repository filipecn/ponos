#ifndef PONOS_STRUCTURES_OBJECT_POOL_H
#define PONOS_STRUCTURES_OBJECT_POOL_H

#include "common/memory.h"

#include <functional>
#include <memory>
#include <stack>
#include <vector>

namespace ponos {

	template<typename ObjectType>
		union PoolElement {
			struct Data {
				ObjectType data;
				PoolElement *next, *prev;
			} element;
			PoolElement *next;
			PoolElement() {}
			~PoolElement() {}
		};

	/* memory
	 * Stores a list of objects with maximum size.
	 */
	template<typename ObjectType, size_t SIZE>
		class ObjectPool {
			typedef PoolElement<ObjectType>* ElementPtr;

			public:
			class iterator {
				public:
					iterator(ObjectPool<ObjectType, SIZE>& op)
						: cur(op.activeHead), pool(op) {}
					bool next() {
						return cur != nullptr;
					}
					ObjectType* get() {
						if(cur == nullptr)
							return nullptr;
						return &cur->element.data;
					}
					ObjectType* operator*() {
						if(cur == nullptr)
							return nullptr;
						return &cur->element.data;
					}
					void operator++() {
						if(cur != nullptr)
							cur = cur->element.next;
					}

				private:
					PoolElement<ObjectType>* cur;
					ObjectPool<ObjectType, SIZE>& pool;
			};

			ObjectPool() {
				for(size_t i = 0; i < SIZE - 1; i++)
					pool[i].next = &pool[i + 1];
				freeHead = &pool[0];
				freeCount = SIZE;
				activeHead = activeTail = nullptr;
			}
			virtual ~ObjectPool() {}

			template<typename... Args>
				ObjectType* create(Args&&... args) {
					if(freeCount == 0)
							return nullptr;
					ElementPtr tmp = freeHead;
					freeHead = freeHead->next;
					freeCount--;
					tmp->element.prev = activeTail;
					tmp->element.next = nullptr;
					if(activeTail != nullptr)
						activeTail->element.next = tmp;
					activeTail = tmp;
					if(activeHead == nullptr)
						activeHead = activeTail;
					new (&activeTail->element.data) ObjectType(std::forward<Args>(args)...);
					return &activeTail->element.data;
				}

			void destroy(ObjectType *obj) {
				size_t ind = (reinterpret_cast<unsigned long long>(obj) -
						reinterpret_cast<unsigned long long>(&pool[0])) / sizeof(PoolElement<ObjectType>);
				if(ind < 0 || ind >= SIZE)
					return;
				ElementPtr prev = pool[ind].element.prev;
				ElementPtr next = pool[ind].element.next;
				if(prev != nullptr)
					prev->element.next = next;
				if(next != nullptr)
					next->element.prev = prev;
				if(activeHead != nullptr)
					activeHead = activeHead->element.next;
				if(activeTail != nullptr)
					activeTail = activeTail->element.prev;
				pool[ind].next = freeHead;
				freeHead = &pool[ind];
				freeCount++;
			}

			size_t size() {
				return SIZE - freeCount;
			}

			private:
			PoolElement<ObjectType> *freeHead;
			PoolElement<ObjectType> *activeHead;
			PoolElement<ObjectType> *activeTail;
			size_t freeCount;
			PoolElement<ObjectType> pool[SIZE];
		};

	template<typename ObjectType>
		union DynamicPoolElement {
			struct Data {
				ObjectType data;
				int next, prev;
			} element;
			int next;
			DynamicPoolElement() {}
		};

	template<typename ObjectType>
		class DynamicObjectPool {
			public:
			class iterator {
				public:
					iterator(DynamicObjectPool<ObjectType>& op)
						: cur(op.activeHead), pool(op) {}
					bool next() {
						return cur != -1;
					}
					ObjectType* get() {
						if(cur == -1)
							return nullptr;
						return &pool.pool[cur].element.data;
					}
					ObjectType* operator*() {
						if(cur == -1)
							return nullptr;
						return &pool.pool[cur].element.data;
					}
					void operator++() {
						if(cur != -1)
							cur = pool.pool[cur].element.next;
					}

				private:
					int cur;
					DynamicObjectPool<ObjectType>& pool;
			};

			DynamicObjectPool() {
				freeHead = -1;
				freeCount = 0;
				activeHead = activeTail = -1;
			}
			virtual ~DynamicObjectPool() {}

			template<typename... Args>
				int create(Args&&... args) {
					if(freeCount == 0) {
						pool.resize(pool.size() + 1);
						freeHead = pool.size() - 1;
						freeCount++;
					}
					int tmp = freeHead;
					freeHead = pool[freeHead].next;
					freeCount--;
					pool[tmp].element.prev = activeTail;
					pool[tmp].element.next = -1;
					if(activeTail != -1)
						pool[activeTail].element.next = tmp;
					activeTail = tmp;
					if(activeHead == -1)
						activeHead = activeTail;
					new (&pool[activeTail].element.data) ObjectType(std::forward<Args>(args)...);
					return activeTail;
				}

			void destroy(int i) {
				if(i < 0 || i >= static_cast<int>(pool.size()))
					return;
				int prev = pool[i].element.prev;
				int next = pool[i].element.next;
				if(prev != -1)
					pool[prev].element.next = next;
				if(next != -1)
					pool[next].element.prev = prev;
				if(activeHead == i)
					activeHead = pool[activeHead].element.next;
				if(activeTail == i)
					activeTail = pool[activeTail].element.prev;
				pool[i].next = freeHead;
				freeHead = i;
				freeCount++;
			}

			size_t size() {
				return pool.size() - freeCount;
			}

			private:
			int freeHead;
			int activeHead;
			int activeTail;
			size_t freeCount;
			std::vector<DynamicPoolElement<ObjectType> > pool;
		};

	template<typename ObjectType>
		class CObjectPool : public IndexPointerInterface<ObjectType> {
			public:
				CObjectPool(size_t size = 0, bool fixed = false) {
					if(size > 0)
						pool.resize(size);
					fixedSize = fixed;
					end = size;
				}
				class iterator {
					public:
						iterator(CObjectPool<ObjectType>& g)
							: cur(0), grid(g) {}
						bool next() {
							return cur < grid.end;
						}
						ObjectType* get() {
							if(cur >= grid.end)
								return nullptr;
							return &grid.pool[cur];
						}
						ObjectType* operator*() {
							if(cur >= grid.end)
								return nullptr;
							return &grid.pool[cur];
						}
						void operator++() {
							cur++;
						}

					private:
						size_t cur;
						CObjectPool<ObjectType>& grid;
				};

				virtual ~CObjectPool() {}
				template<typename... Args>
					ponos::IndexPointer<ObjectType> create(Args&&... args) {
						if(static_cast<size_t>(end) == pool.size())
							pool.emplace_back(std::forward<Args>(args)...);
						else
							new (&pool[end]) ObjectType(std::forward<Args>(args)...);
						indexTable.push_back(end);
						if(!freeIndices.empty()) {
							size_t t = freeIndices.top();
							freeIndices.pop();
							indexTable[t] = end++;
							return ponos::IndexPointer<ObjectType>(t, this);
						}
						return ponos::IndexPointer<ObjectType>(end++, this);
					}
				virtual void destroy(ponos::IndexPointer<ObjectType> i) {
					if(i.getIndex() >= indexTable.size())
						return;
					size_t ind = indexTable[i.getIndex()];
					if(ind >= end || end <= 0)
						return;
					pool[ind] = pool[end - 1];
					indexTable[end - 1] = ind;
					end--;
					freeIndices.push(i.getIndex());
				}
				size_t size() const { return end; }
				ObjectType* get(size_t i) override {
					if(i >= end || i >= indexTable.size())
						return nullptr;
					return &pool[indexTable[i]];
				}

			protected:
				size_t end;
				bool fixedSize;
				std::vector<ObjectType> pool;
				std::vector<size_t> indexTable;
				std::stack<size_t> freeIndices;
		};

} // odysseus namespace

#endif // PONOS_STRUCTURES_OBJECT_POOL_H

