#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Array1, Constructors) {
  {
    Array1<vec2> a(1000);
    EXPECT_EQ(a.size(), 1000u);
  }
  {
    Array1<vec2i> a(1000, vec2i(1, 3));
    auto v = a.hostData();
    for (auto vv : v)
      EXPECT_EQ(vv, vec2i(1, 3));
  }
  {
    Array1<char> a(1000);
    a = 'a';
    auto v = a.hostData();
    for (auto c : v)
      EXPECT_EQ(c, 'a');
  }
  {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i)
      data[i] = i;
    Array1<int> a = data;
    auto v = a.hostData();
    for (int i = 0; i < 1000; ++i)
      EXPECT_EQ(v[i], i);
  }
  {
    Array1<int> a = std::move(Array1<int>(1000));
    Array1<int> b(Array1<int>(1000));
    b = a;
    a = std::vector<int>(1000);
  }
}

TEST(Array2, Sanity) {
  {
    Array2<vec2> a(size2(10, 10));
    EXPECT_EQ(a.size(), size2(10, 10));
    EXPECT_EQ(a.memorySize(), 10 * a.pitch());
    Array2<vec2> b = a;
    EXPECT_EQ(b.size(), size2(10, 10));
    EXPECT_EQ(b.memorySize(), 10 * b.pitch());
  }
  {
    Array2<int> a(size2(10, 10));
    a = 3;
    auto ha = a.hostData();
    int count = 0;
    for (auto e : ha) {
      EXPECT_EQ(e.value, 3);
      count++;
    }
    EXPECT_EQ(count, 10 * 10);
  }
  {
    std::vector<Array2<int>> v;
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    std::vector<ponos::Array2<int>> h_v;
    h_v.emplace_back(ponos::size2(10, 10));
    h_v.emplace_back(ponos::size2(10, 10));
    h_v.emplace_back(ponos::size2(10, 10));
    for (int i = 0; i < 3; i++) {
      for (auto e : h_v[i])
        e.value = e.index.i * 10 + e.index.j;
      v[i] = h_v[i];
    }
    std::vector<Array2<int>> vv;
    for (auto &e : v)
      vv.emplace_back(e);
    for (int i = 0; i < 3; i++) {
      auto h = vv[i].hostData();
      int count = 0;
      for (auto hh : h) {
        EXPECT_EQ(hh.value, hh.index.i * 10 + hh.index.j);
        count++;
      }
      EXPECT_EQ(count, 100);
    }
  }
  {
    Array2<int> a = std::move(Array2<int>(size2(10, 10)));
    Array2<int> b(Array2<int>(size2(10, 10)));
  }
}

template <typename T> struct map_ipj {
  __host__ __device__ void operator()(index2 index, T &value) const {
    value = index.i + index.j;
  }
};

TEST(Array2, Methods) {
  {
    // array2i a(size2(10));
    // a.map(map_ipj<int>());
  }
}

TEST(MemoryBlock1, Constructors) {
  { CuMemoryBlock1<vec2> a(1000); }
  {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i)
      data[i] = i;
    CuMemoryBlock1<int> a = data;
    auto v = a.hostData();
    for (int i = 0; i < 1000; ++i)
      EXPECT_EQ(v[i], i);
  }
  {
    CuMemoryBlock1<int> a = std::move(CuMemoryBlock1<int>(1000));
    CuMemoryBlock1<int> b(CuMemoryBlock1<int>(1000));
    b = a;
    a = std::vector<int>(1000);
  }
}

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