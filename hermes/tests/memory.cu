#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(MemoryBlock2, memcpy) {
  {
    vec2u res(8);
    MemoryBlock2<MemoryLocation::HOST, int> lmb(res);
    lmb.allocate();
    {
      auto acc = lmb.accessor();
      for (size_t j = 0; j < res.y; j++)
        for (size_t i = 0; i < res.x; i++)
          acc(i, j) = i * 100 + j * 10;
    }
    MemoryBlock2<MemoryLocation::DEVICE, int> dlmb(res);
    dlmb.allocate();
    memcpy(dlmb, lmb);
    MemoryBlock2<MemoryLocation::HOST, int> lmb2(res);
    lmb2.allocate();
    memcpy(lmb2, dlmb);
    {
      auto acc = lmb2.accessor();
      for (size_t j = 0; j < res.y; j++)
        for (size_t i = 0; i < res.x; i++)
          EXPECT_EQ(acc(i, j), i * 100 + j * 10);
    }
  }
}

TEST(MemoryBlock3, memcpy) {
  {
    vec3u res(8);
    MemoryBlock3<MemoryLocation::HOST, int> lmb(res);
    lmb.allocate();
    {
      auto acc = lmb.accessor();
      for (size_t k = 0; k < res.z; k++)
        for (size_t j = 0; j < res.y; j++)
          for (size_t i = 0; i < res.x; i++)
            acc(i, j, k) = i * 100 + j * 10 + k;
    }
    MemoryBlock3<MemoryLocation::DEVICE, int> dlmb(res);
    dlmb.allocate();
    memcpy(dlmb, lmb);
    MemoryBlock3<MemoryLocation::HOST, int> lmb2(res);
    lmb2.allocate();
    memcpy(lmb2, dlmb);
    {
      auto acc = lmb2.accessor();
      for (size_t k = 0; k < res.z; k++)
        for (size_t j = 0; j < res.y; j++)
          for (size_t i = 0; i < res.x; i++)
            EXPECT_EQ(acc(i, j, k), i * 100 + j * 10 + k);
    }
  }
}

TEST(MemoryBlock2, iterator) {
  MemoryBlock2Hi m(vec2u(5));
  m.allocate();
  auto acc = m.accessor();
  for (auto e : acc)
    e.value = e.j() * 10 + e.i();
  for (int j = 0; j < 5; j++)
    for (int i = 0; i < 5; i++)
      EXPECT_EQ(acc(i, j), j * 10 + i);
}

TEST(MemoryBlock3, iterator) {
  MemoryBlock3Hi m(vec3u(5));
  m.allocate();
  auto acc = m.accessor();
  for (auto e : acc)
    e.value = e.k() * 100 + e.j() * 10 + e.i();
  for (int k = 0; k < 5; k++)
    for (int j = 0; j < 5; j++)
      for (int i = 0; i < 5; i++)
        EXPECT_EQ(acc(i, j, k), k * 100 + j * 10 + i);
}