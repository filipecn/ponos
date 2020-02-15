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
    EXPECT_EQ(point3f(0.f),
              acc.uAccessor().gridPosition(point3f(-0.5f, 0.f, 0.f)));
    EXPECT_EQ(point3f(0.f),
              acc.vAccessor().gridPosition(point3f(0.f, -0.5f, 0.f)));
    EXPECT_EQ(point3f(0.f),
              acc.wAccessor().gridPosition(point3f(0.f, 0.f, -0.5f)));
    EXPECT_EQ(point3f(-0.5f, 0.f, 0.f), acc.uAccessor().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, -0.5f, 0.f), acc.vAccessor().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, 0.f, -0.5f), acc.wAccessor().worldPosition(0, 0, 0));
  }
  {
    StaggeredGrid3H sg;
    sg.setSpacing(vec3f(0.1));
    auto acc = sg.accessor();
    EXPECT_EQ(point3f(0.f),
              acc.uAccessor().gridPosition(point3f(-0.05f, 0.f, 0.f)));
    EXPECT_EQ(point3f(0.f),
              acc.vAccessor().gridPosition(point3f(0.f, -0.05f, 0.f)));
    EXPECT_EQ(point3f(0.f),
              acc.wAccessor().gridPosition(point3f(0.f, 0.f, -0.05f)));
    EXPECT_EQ(point3f(-0.05f, 0.f, 0.f),
              acc.uAccessor().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, -0.05f, 0.f),
              acc.vAccessor().worldPosition(0, 0, 0));
    EXPECT_EQ(point3f(0.f, 0.f, -0.05f),
              acc.wAccessor().worldPosition(0, 0, 0));
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

TEST(RegularGrid2, iterator) {
  RegularGrid2Hi m(vec2u(5));
  m.setSpacing(vec2f(0.1));
  auto acc = m.accessor();
  for (auto e : acc) {
    e.value = e.j() * 10 + e.i();
    EXPECT_EQ(point2f(e.i() * 0.1f, e.j() * 0.1f), e.worldPosition());
  }
  for (int j = 0; j < 5; j++)
    for (int i = 0; i < 5; i++)
      EXPECT_EQ(acc(i, j), j * 10 + i);
}

TEST(RegularGrid3, iterator) {
  RegularGrid3Hi m(vec3u(5));
  m.setSpacing(vec3f(0.1));
  auto acc = m.accessor();
  for (auto e : acc) {
    e.value = e.k() * 100 + e.j() * 10 + e.i();
    EXPECT_EQ(point3f(e.i() * 0.1f, e.j() * 0.1f, e.k() * 0.1f),
              e.worldPosition());
  }
  for (int k = 0; k < 5; k++)
    for (int j = 0; j < 5; j++)
      for (int i = 0; i < 5; i++)
        EXPECT_EQ(acc(i, j, k), k * 100 + j * 10 + i);
}

TEST(RegularGrid2, reduce) {
  {
    vec2u res(100);
    RegularGrid2Hf g(100);
    for (auto e : g.accessor())
      e.value = e.i() - e.j();
    RegularGrid2Df d_g(g);
    EXPECT_EQ(minValue(d_g), -99);
    EXPECT_EQ(maxValue(d_g), 99);
  }
}