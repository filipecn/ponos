#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(StaggeredGridTexture, transform) {
  StaggeredGridTexture2 sg;
  EXPECT_EQ(point2f(0.f, 0.f), sg.u().toFieldTransform()(point2f(-0.5f, 0.f)));
  EXPECT_EQ(point2f(0.f, 0.f), sg.v().toFieldTransform()(point2f(0.f, -0.5f)));
  EXPECT_EQ(point2f(-0.5f, 0.0f), sg.u().toWorldTransform()(point2f(0.f, 0.f)));
  EXPECT_EQ(point2f(0.0f, -0.5f), sg.v().toWorldTransform()(point2f(0.f, 0.f)));
}

TEST(StaggeredGrid3, transform) {
  {
    StaggeredGrid3H sg;
    auto acc = sg.accessor();
    EXPECT_EQ(point3f(0.f), acc.u().gridPosition(point3f(-0.5f, 0.f, 0.f)));
    EXPECT_EQ(point3f(0.f), acc.v().gridPosition(point3f(0.f, -0.5f, 0.f)));
    EXPECT_EQ(point3f(0.f), acc.w().gridPosition(point3f(0.f, 0.f, -0.5f)));
    EXPECT_EQ(point3f(-0.5f, 0.f, 0.f), acc.u().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, -0.5f, 0.f), acc.v().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, 0.f, -0.5f), acc.w().worldPosition(0, 0, 0));
  }
  {
    StaggeredGrid3H sg;
    sg.setSpacing(vec3f(0.1));
    auto acc = sg.accessor();
    EXPECT_EQ(point3f(0.f), acc.u().gridPosition(point3f(-0.05f, 0.f, 0.f)));
    EXPECT_EQ(point3f(0.f), acc.v().gridPosition(point3f(0.f, -0.05f, 0.f)));
    EXPECT_EQ(point3f(0.f), acc.w().gridPosition(point3f(0.f, 0.f, -0.05f)));
    EXPECT_EQ(point3f(-0.05f, 0.f, 0.f), acc.u().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, -0.05f, 0.f), acc.v().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, 0.f, -0.05f), acc.w().worldPosition(0, 0, 0));
  }
}

TEST(RegularGrid, accessor) {
  {
    vec3u res(128);
    RegularGrid3Hi grid(res);
    grid.setSpacing(vec3f(0.01));
    auto acc = grid.accessor();
    int idx = 0;
    for (int k = 0; k < res.z; k++)
      for (int j = 0; j < res.y; j++)
        for (int i = 0; i < res.x; i++)
          acc(i, j, k) = idx++;
  }
}
