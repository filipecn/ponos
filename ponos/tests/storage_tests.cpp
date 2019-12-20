#include <gtest/gtest.h>

#include <ponos/ponos.h>

using namespace ponos;

TEST(MemoryBlock2, Constructors) {
  {
    MemoryBlock2<vec2> a(size2(10, 10));
    for (index2 ij : Index2Range<i32>(a.size()))
      a(ij) = vec2(ij.i, ij.j);
    MemoryBlock2<vec2> b = a;
    for (index2 ij : Index2Range<i32>(a.size()))
      EXPECT_EQ(a(ij), b(ij));
  }
  {
    std::vector<MemoryBlock2<int>> v;
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    v.emplace_back(size2(10, 10));
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        v[i](ij) = ij.i * 10 + ij.j;
    std::vector<MemoryBlock2<int>> vv = v;
    for (int i = 0; i < 3; i++)
      for (index2 ij : Index2Range<i32>(v[i].size()))
        EXPECT_EQ(vv[i](ij), ij.i * 10 + ij.j);
  }
  {
    MemoryBlock2<int> a = std::move(MemoryBlock2<int>(size2(10, 10)));
    MemoryBlock2<int> b(MemoryBlock2<int>(size2(10, 10)));
  }
}

TEST(MemoryBlock2, Methods) {
  { // copy
    MemoryBlock2<int> a(size2(10, 10));
    for (index2 ij : Index2Range<i32>(a.size()))
      a(ij) = ij.i * 10 + ij.j;
    MemoryBlock2<int> b;
    b.copy(a);
    EXPECT_EQ(a.pitch(), b.pitch());
    EXPECT_EQ(a.size(), b.size());
    for (index2 ij : Index2Range<i32>(a.size()))
      EXPECT_EQ(a(ij), b(ij));
  }
  { // value assign
    MemoryBlock2<int> a(size2(10, 10));
    a = 3;
    for (index2 ij : Index2Range<i32>(a.size()))
      EXPECT_EQ(a(ij), 3);
  }
}

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
