//
// Created by filipecn on 2018-12-20.
//

#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(BBox3, Access) {
  {
    auto l = point3(0, 1, 2);
    auto u = point3(3, 4, 3);
    bbox3 b(l, u);
    EXPECT_TRUE(b.lower == l);
    EXPECT_TRUE(b.upper == u);
    EXPECT_TRUE(b[0] == l);
    EXPECT_TRUE(b[1] == u);
    EXPECT_TRUE(b.corner(0) == point3(0, 1, 2));
    EXPECT_TRUE(b.corner(1) == point3(3, 1, 2));
    EXPECT_TRUE(b.corner(2) == point3(0, 4, 2));
    EXPECT_TRUE(b.corner(3) == point3(3, 4, 2));
    EXPECT_TRUE(b.corner(4) == point3(0, 1, 3));
    EXPECT_TRUE(b.corner(5) == point3(3, 1, 3));
    EXPECT_TRUE(b.corner(6) == point3(0, 4, 3));
    EXPECT_TRUE(b.corner(7) == point3(3, 4, 3));
  }
}

TEST(BBox3, Union) {
  {
    bbox3 a(point3(), point3(1));
    bbox3 b(point3(-1), point3());
    bbox3 c = make_union(a, b);
    point3 l(-1), u(1);
    EXPECT_TRUE(c.lower == l);
    EXPECT_TRUE(c.upper == u);
  }
}