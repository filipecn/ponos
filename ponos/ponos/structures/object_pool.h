#ifndef PONOS_STRUCTURES_OBJECT_POOL_H
#define PONOS_STRUCTURES_OBJECT_POOL_H

#include <ponos/storage/memory.h>

#include <functional>
#include <memory>
#include <stack>
#include <vector>

namespace ponos {

template <typename ObjectType> union PoolElement {
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
template <typename ObjectType, size_t SIZE> class ObjectPool {
  typedef PoolElement<ObjectType> *ElementPtr;

public:
  class iterator {
  public:
    iterator(ObjectPool<ObjectType, SIZE> &op) : cur(op.activeHead), pool(op) {}
    bool next() { return cur != nullptr; }
    ObjectType *get() {
      if (cur == nullptr)
        return nullptr;
      return &cur->element.data;
    }
    ObjectType *operator*() {
      if (cur == nullptr)
        return nullptr;
      return &cur->element.data;
    }
    void operator++() {
      if (cur != nullptr)
        cur = cur->element.next;
    }

  private:
    PoolElement<ObjectType> *cur;
    ObjectPool<ObjectType, SIZE> &pool;
  };

  ObjectPool() {
    for (size_t i = 0; i < SIZE - 1; i++)
      pool[i].next = &pool[i + 1];
    freeHead = &pool[0];
    freeCount = SIZE;
    activeHead = activeTail = nullptr;
  }
  virtual ~ObjectPool() {}

  template <typename... Args> ObjectType *create(Args &&... args) {
    if (freeCount == 0)
      return nullptr;
    ElementPtr tmp = freeHead;
    freeHead = freeHead->next;
    freeCount--;
    tmp->element.prev = activeTail;
    tmp->element.next = nullptr;
    if (activeTail != nullptr)
      activeTail->element.next = tmp;
    activeTail = tmp;
    if (activeHead == nullptr)
      activeHead = activeTail;
    new (&activeTail->element.data) ObjectType(std::forward<Args>(args)...);
    return &activeTail->element.data;
  }

  void destroy(ObjectType *obj) {
    size_t ind = (reinterpret_cast<unsigned long long>(obj) -
                  reinterpret_cast<unsigned long long>(&pool[0])) /
                 sizeof(PoolElement<ObjectType>);
    if (ind < 0 || ind >= SIZE)
      return;
    ElementPtr prev = pool[ind].element.prev;
    ElementPtr next = pool[ind].element.next;
    if (prev != nullptr)
      prev->element.next = next;
    if (next != nullptr)
      next->element.prev = prev;
    if (activeHead != nullptr)
      activeHead = activeHead->element.next;
    if (activeTail != nullptr)
      activeTail = activeTail->element.prev;
    pool[ind].next = freeHead;
    freeHead = &pool[ind];
    freeCount++;
  }

  size_t size() { return SIZE - freeCount; }

private:
  PoolElement<ObjectType> *freeHead;
  PoolElement<ObjectType> *activeHead;
  PoolElement<ObjectType> *activeTail;
  size_t freeCount;
  PoolElement<ObjectType> pool[SIZE];
};

template <typename ObjectType> union DynamicPoolElement {
  struct Data {
    ObjectType data;
    int next, prev;
  } element;
  int next;
  DynamicPoolElement() {}
};

template <typename ObjectType> class DynamicObjectPool {
public:
  class iterator {
  public:
    iterator(DynamicObjectPool<ObjectType> &op)
        : cur(op.activeHead), pool(op) {}
    bool next() { return cur != -1; }
    ObjectType *get() {
      if (cur == -1)
        return nullptr;
      return &pool.pool[cur].element.data;
    }
    ObjectType *operator*() {
      if (cur == -1)
        return nullptr;
      return &pool.pool[cur].element.data;
    }
    void operator++() {
      if (cur != -1)
        cur = pool.pool[cur].element.next;
    }

  private:
    int cur;
    DynamicObjectPool<ObjectType> &pool;
  };

  DynamicObjectPool() {
    freeHead = -1;
    freeCount = 0;
    activeHead = activeTail = -1;
  }
  virtual ~DynamicObjectPool() {}

  template <typename... Args> int create(Args &&... args) {
    if (freeCount == 0) {
      pool.resize(pool.size() + 1);
      freeHead = pool.size() - 1;
      freeCount++;
    }
    int tmp = freeHead;
    freeHead = pool[freeHead].next;
    freeCount--;
    pool[tmp].element.prev = activeTail;
    pool[tmp].element.next = -1;
    if (activeTail != -1)
      pool[activeTail].element.next = tmp;
    activeTail = tmp;
    if (activeHead == -1)
      activeHead = activeTail;
    new (&pool[activeTail].element.data)
        ObjectType(std::forward<Args>(args)...);
    return activeTail;
  }

  void destroy(int i) {
    if (i < 0 || i >= static_cast<int>(pool.size()))
      return;
    int prev = pool[i].element.prev;
    int next = pool[i].element.next;
    if (prev != -1)
      pool[prev].element.next = next;
    if (next != -1)
      pool[next].element.prev = prev;
    if (activeHead == i)
      activeHead = pool[activeHead].element.next;
    if (activeTail == i)
      activeTail = pool[activeTail].element.prev;
    pool[i].next = freeHead;
    freeHead = i;
    freeCount++;
  }

  size_t size() { return pool.size() - freeCount; }

private:
  int freeHead;
  int activeHead;
  int activeTail;
  size_t freeCount;
  std::vector<DynamicPoolElement<ObjectType>> pool;
};

template <typename ObjectType>
class CObjectPool : public IndexPointerInterface<ObjectType> {
public:
  CObjectPool(size_t size = 0, bool fixed = false) {
    if (size > 0)
      pool.resize(size);
    fixedSize = fixed;
    end = size;
  }
  class iterator {
  public:
    iterator(CObjectPool<ObjectType> &g) : cur(0), grid(g) {}
    bool next() { return cur < grid.end; }
    ObjectType *get() {
      if (cur >= grid.end)
        return nullptr;
      return &grid.pool[cur];
    }
    ObjectType *operator*() {
      if (cur >= grid.end)
        return nullptr;
      return &grid.pool[cur];
    }
    void operator++() { cur++; }

  private:
    size_t cur;
    CObjectPool<ObjectType> &grid;
  };

  virtual ~CObjectPool() {}
  template <typename... Args>
  ponos::IndexPointer<ObjectType> create(Args &&... args) {
    if (end == pool.size()) {
      pool.emplace_back(std::forward<Args>(args)...);
    } else
      new (&pool[end]) ObjectType(std::forward<Args>(args)...);
    if (!freeIndices.empty()) {
      size_t t = freeIndices.top();
      freeIndices.pop();
      indexToPool[t] = end;
      poolToIndex[end] = t;
      end++;
      return IndexPointer<ObjectType>(t, this);
    }
    indexToPool.push_back(end);
    poolToIndex.push_back(end);
    return IndexPointer<ObjectType>(end++, this);
  }
  virtual void destroy(IndexPointer<ObjectType> i) { destroy(i.getIndex()); }
  virtual void destroy(size_t elementIndex) {
    if (elementIndex >= indexToPool.size())
      return;

    int poolIndex = indexToPool[elementIndex];
    int lastPoolIndex = end - 1;

    if (poolIndex >= static_cast<int>(end) || poolIndex < 0 ||
        lastPoolIndex < 0)
      return;
    // std::cout << "destroying " << element_index << std::endl;
    pool[poolIndex].destroy();
    pool[poolIndex] = pool[lastPoolIndex];

    int lastElementIndex = poolToIndex[lastPoolIndex];

    indexToPool[elementIndex] = -1;
    indexToPool[lastElementIndex] =
        (static_cast<int>(elementIndex) == lastElementIndex) ? -1 : poolIndex;

    poolToIndex[poolIndex] =
        (static_cast<int>(elementIndex) == lastElementIndex) ? -1
                                                             : lastElementIndex;
    poolToIndex[lastPoolIndex] = -1;

    end--;
    freeIndices.push(elementIndex);
  }
  size_t size() const {
    std::cout << "size " << end << std::endl;
    std::cout << "index to pool" << std::endl;
    for (size_t i = 0; i < indexToPool.size(); i++)
      std::cout << i << " | " << indexToPool[i] << std::endl;
    std::cout << "pool to index" << std::endl;
    for (size_t i = 0; i < poolToIndex.size(); i++)
      std::cout << i << " | " << poolToIndex[i] << std::endl;
    std::cout << std::endl;
    return end;
  }
  ObjectType *get(size_t i) override {
    if (i >= end || i >= indexToPool.size() || indexToPool[i] < 0)
      return nullptr;
    return &pool[indexToPool[i]];
  }

protected:
  size_t end;
  bool fixedSize;
  std::vector<ObjectType> pool;
  std::vector<int> indexToPool, poolToIndex;
  std::stack<size_t> freeIndices;
};

} // odysseus namespace

#endif // PONOS_STRUCTURES_OBJECT_POOL_H
