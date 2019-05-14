#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(MemoryBlock3, memcpy) {
  { // linear - linear
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