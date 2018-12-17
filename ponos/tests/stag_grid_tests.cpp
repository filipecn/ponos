#include <gtest/gtest.h>

#include <ponos/ponos.h>

using namespace ponos;

TEST(StaggeredGrid2f, Access) {
  float error = 1e-8;
  StaggeredGrid2f grid;
  grid.set(10, 10, BBox2(Point2(), Point2(1, 1)));
  { // TEST DIMENSIONS
    EXPECT_EQ(grid.p.width, 10u);
    EXPECT_EQ(grid.p.height, 10u);
    EXPECT_EQ(grid.u.width, 11u);
    EXPECT_EQ(grid.u.height, 10u);
    EXPECT_EQ(grid.v.width, 10u);
    EXPECT_EQ(grid.v.height, 11u);
  }
  { // TEST WORLD POSITION
    Point2 wp = grid.p.dataWorldPosition(5, 5);
    ASSERT_NEAR(wp.x, 0.55f, error);
    ASSERT_NEAR(wp.y, 0.55f, error);
    wp = grid.u.dataWorldPosition(5, 5);
    ASSERT_NEAR(wp.x, 0.5f, error);
    ASSERT_NEAR(wp.y, 0.55f, error);
    wp = grid.v.dataWorldPosition(5, 5);
    ASSERT_NEAR(wp.x, 0.55f, error);
    ASSERT_NEAR(wp.y, 0.5f, error);
  }
  { // TEST ACCESS
    EXPECT_EQ(grid.p.accessMode, GridAccessMode::RESTRICT);
    grid.p.setAll(1.f);
    grid.u.setAll(1.f);
    grid.v.setAll(1.f);
    try {
      grid.p(1000, 1000);
      EXPECT_EQ(1, 0);
    } catch (const char *m) {
      EXPECT_EQ(1, 1);
    }
    EXPECT_EQ(grid.u.accessMode, GridAccessMode::CLAMP_TO_EDGE);
    EXPECT_EQ(grid.v.accessMode, GridAccessMode::CLAMP_TO_EDGE);
    ASSERT_NEAR(grid.u(1000, 1000), 1.f, error);
    ASSERT_NEAR(grid.v(1000, 1000), 1.f, error);
  }
}

TEST(StaggeredGrid2f, Sample) {}
