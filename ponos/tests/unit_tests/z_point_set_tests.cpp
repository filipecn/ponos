#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(ZPointSet, Constructors) {
  {
    ZPointSet s;
    EXPECT_EQ(s.size(), 0u);
  }
  {
    ZPointSet s(4);
    EXPECT_EQ(s.size(), 0u);
  }
}

TEST(ZPointSet, Iterator) {
  {
    ZPointSet z(2u);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          z.add(Point3(1.f * i, 1.f * j, 1.f * k));
    ZPointSet::iterator it(z);
    int i = 0, j = 0, k = 0;
    uint count = 0;
    while (it.next()) {
      EXPECT_EQ(Point3(1. * i, 1. * j, 1. * k), it.getWorldPosition());
      EXPECT_EQ(count++, it.getId());
      k++;
      if (k >= 2) {
        k = 0;
        j++;
        if (j >= 2) {
          j = 0;
          i++;
        }
      }
      auto wp = it.getWorldPosition();
      EXPECT_EQ(
          it.pointElement()->zcode,
          mortonCode(wp.x, wp.y, wp.z));
      ++it;
    }
    EXPECT_EQ(count, z.size());
  }
}