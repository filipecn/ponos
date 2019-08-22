//
// Created by filipecn on 2018-12-20.
//

#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(Common, Index) {
  { // Index2
    index2 a;
    index2 b;
    EXPECT_EQ(a, b);
    b.j = 1;
    EXPECT_NE(a, b);
  }
  { // Index2Range
    int cur = 0;
    for (auto index : Index2Range<i32>(10, 10)) {
      EXPECT_EQ(cur % 10, index.i);
      EXPECT_EQ(cur / 10, index.j);
      cur++;
    }
    EXPECT_EQ(cur, 10 * 10);
  }
  { // Index3
    index3 a;
    index3 b;
    EXPECT_EQ(a, b);
    b.j = 1;
    EXPECT_NE(a, b);
  }
  { // Index3Range
    int cur = 0;
    for (auto index : Index3Range<i32>(10, 10, 10)) {
      EXPECT_EQ((cur % 100) % 10, index.i);
      EXPECT_EQ((cur % 100) / 10, index.j);
      EXPECT_EQ(cur / 100, index.k);
      cur++;
    }
    EXPECT_EQ(cur, 10 * 10 * 10);
  }
}