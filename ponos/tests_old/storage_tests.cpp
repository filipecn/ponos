#include <gtest/gtest.h>

#include <ponos/ponos.h>

using namespace ponos;

TEST(Array2, Sanity) {
  {
    Array2<vec2> a(size2(10, 10));
    EXPECT_EQ(a.pitch(), 10 * sizeof(vec2));
    EXPECT_EQ(a.size(), size2(10, 10));
    EXPECT_EQ(a.memorySize(), 10 * 10 * sizeof(vec2));
    for (index2 ij : Index2Range<i32>(a.size()))
      a[ij] = vec2(ij.i, ij.j);
    Array2<vec2> b = a;
    for (index2 ij : Index2Range<i32>(a.size()))
      EXPECT_EQ(a[ij], b[ij]);
  }
  {
    Array2<int> a(size2(10, 10));
    a = 3;
    int count = 0;
    for (index2 ij : Index2Range<i32>(a.size()))
      EXPECT_EQ(a[ij], 3);
    for (const auto &e : a) {
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
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        v[i][ij] = ij.i * 10 + ij.j;
    std::vector<Array2<int>> vv = v;
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        EXPECT_EQ(vv[i][ij], ij.i * 10 + ij.j);
  }
  {
    Array2<int> a = std::move(Array2<int>(size2(10, 10)));
    Array2<int> b(Array2<int>(size2(10, 10)));
  }
}

TEST(Array, iterator) {
  {
    Array2<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      EXPECT_EQ(e.value, vec2(1, 2));
    }
    EXPECT_EQ(count, 100);
  }
}

/*
TEST(MemoryBlock2, Access) {
  { // no pitched size
    MemoryBlock2<int> m(size2(10, 10));
    EXPECT_EQ(m.pitch(), 10 * sizeof(int));
    EXPECT_EQ(m.size(), size2(10, 10));
    EXPECT_EQ(m.memorySize(), 10 * 10 * sizeof(int));
    for (index2 ij : Index2Range<i32>(m.size())) {
      m(ij) = ij.i * 10 + ij.j;
      EXPECT_EQ(m(ij), ij.i * 10 + ij.j);
    }
  }
  { // pitched size
    MemoryBlock2<int> m(size2(10, 10), sizeof(int) * 12);
    EXPECT_EQ(m.pitch(), 12 * sizeof(int));
    EXPECT_EQ(m.size(), size2(10, 10));
    EXPECT_EQ(m.memorySize(), 12 * 10 * sizeof(int));
    for (index2 ij : Index2Range<i32>(m.size())) {
      m(ij) = ij.i * 10 + ij.j;
      EXPECT_EQ(m(ij), ij.i * 10 + ij.j);
    }
  }
}
*/