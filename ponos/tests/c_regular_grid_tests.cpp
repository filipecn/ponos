#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(ScalarGrid2f, Access) {
  float error = 1e-8;
  ScalarGrid2f grid(10, 10);
  grid.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  grid.setAll(1.f);
  int count = 0;
  grid.forEach([&](float &f, int i, int j) {
    UNUSED_VARIABLE(f);
    count++;
    ASSERT_NEAR(grid(i, j), 1.f, error);
    ASSERT_NEAR(grid(ivec2(i, j)), 1.f, error);
  });
  EXPECT_EQ(count, 10 * 10);
}

TEST(ScalarGrid2f, DataPosition) {
  float error = 1e-6;
  ScalarGrid2f grid;
  grid.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  grid.set(10, 10, vec2(0.f, 0.f), vec2(0.1, 0.2));
  { // TEST OFFSET AND WORLD POSITION
    grid.dataPosition = GridDataPosition::VERTEX_CENTER;
    vec2 d = grid.dataOffset();
    ASSERT_NEAR(d[0], 0.0f, error);
    ASSERT_NEAR(d[1], 0.0f, error);
    Point2 wp = grid.dataWorldPosition(ivec2(10, 10));
    ASSERT_NEAR(wp[0], 1.0f, error);
    ASSERT_NEAR(wp[1], 2.0f, error);
    wp = Point2(0.1, 0.2);
    Point2 gp = grid.toGrid(wp);
    ASSERT_NEAR(gp[0], 1.f, error);
    ASSERT_NEAR(gp[1], 1.f, error);
    Point2 dgp = grid.dataGridPosition(wp);
    ASSERT_NEAR(dgp[0], 1.f, error);
    ASSERT_NEAR(dgp[1], 1.f, error);
    grid.dataPosition = GridDataPosition::CELL_CENTER;
    d = grid.dataOffset();
    ASSERT_NEAR(d[0], 0.5f, error);
    ASSERT_NEAR(d[1], 0.5f, error);
    wp = grid.dataWorldPosition(ivec2(10, 10));
    ASSERT_NEAR(wp[0], 1.05f, error);
    ASSERT_NEAR(wp[1], 2.1f, error);
    wp = Point2(0.1, 0.2);
    dgp = grid.dataGridPosition(wp);
    ASSERT_NEAR(dgp[0], 0.5f, error);
    ASSERT_NEAR(dgp[1], 0.5f, error);
    grid.dataPosition = GridDataPosition::U_FACE_CENTER;
    d = grid.dataOffset();
    ASSERT_NEAR(d[0], 0.0f, error);
    ASSERT_NEAR(d[1], 0.5f, error);
    wp = grid.dataWorldPosition(ivec2(10, 10));
    ASSERT_NEAR(wp[0], 1.0f, error);
    ASSERT_NEAR(wp[1], 2.1f, error);
    wp = Point2(0.1, 0.2);
    dgp = grid.dataGridPosition(wp);
    ASSERT_NEAR(dgp[0], 1.f, error);
    ASSERT_NEAR(dgp[1], 0.5f, error);
    grid.dataPosition = GridDataPosition::V_FACE_CENTER;
    d = grid.dataOffset();
    ASSERT_NEAR(d[0], 0.5f, error);
    ASSERT_NEAR(d[1], 0.0f, error);
    wp = grid.dataWorldPosition(ivec2(10, 10));
    ASSERT_NEAR(wp[0], 1.05f, error);
    ASSERT_NEAR(wp[1], 2.0f, error);
    wp = Point2(0.1, 0.2);
    dgp = grid.dataGridPosition(wp);
    ASSERT_NEAR(dgp[0], 0.5f, error);
    ASSERT_NEAR(dgp[1], 1.f, error);
  }
  { // TEST BBOX
    GridDataPosition types[4] = {
        GridDataPosition::CELL_CENTER, GridDataPosition::VERTEX_CENTER,
        GridDataPosition::V_FACE_CENTER, GridDataPosition::U_FACE_CENTER};
    for (auto type : types) {
      grid.dataPosition = type;
      BBox2 b = grid.cellWBox(Point<int, 2>({5, 5}));
      ASSERT_NEAR(b.lower.x, 0.5f, error);
      ASSERT_NEAR(b.lower.y, 1.f, error);
      ASSERT_NEAR(b.upper.x, 0.6f, error);
      ASSERT_NEAR(b.upper.y, 1.2f, error);
    }
  }
}

TEST(ScalarGrid2f, ForEach) {
  ScalarGrid2f grid;
  grid.accessMode = GridAccessMode::CLAMP_TO_EDGE;
  grid.dataPosition = GridDataPosition::CELL_CENTER;
  ImplicitCircle s(Point2(), 0.5f);
  grid.set(15, 15, scale(2, 2)(s.boundingBox()));
  grid.forEach([&](float &f, int i, int j) {
    f = s.signedDistance(grid.dataWorldPosition(i, j));
  });
  // std::cout << &grid;
}

TEST(ScalarGrid2f, DataCell) {
  ScalarGrid2f grid;
  grid.setTransform(scale(0.1, 0.2));
  Point2 wp(0.11, 0.21);
  { // VERTEX CENTER
    grid.dataPosition = GridDataPosition::VERTEX_CENTER;
    Point2i cp = grid.cell(wp);
    Point2i dp = grid.dataCell(wp);
    EXPECT_EQ(cp[0], 1);
    EXPECT_EQ(cp[1], 1);
    EXPECT_EQ(cp[0], dp[0]);
    EXPECT_EQ(cp[1], dp[1]);
  }
  { // CELL CENTER
    grid.dataPosition = GridDataPosition::CELL_CENTER;
    Point2i cp = grid.cell(wp);
    Point2i dp = grid.dataCell(wp);
    EXPECT_EQ(cp[0], 1);
    EXPECT_EQ(cp[1], 1);
    EXPECT_EQ(0, dp[0]);
    EXPECT_EQ(0, dp[1]);
  }
  { // U CENTER
    grid.dataPosition = GridDataPosition::U_FACE_CENTER;
    Point2i cp = grid.cell(wp);
    Point2i dp = grid.dataCell(wp);
    EXPECT_EQ(cp[0], 1);
    EXPECT_EQ(cp[1], 1);
    EXPECT_EQ(1, dp[0]);
    EXPECT_EQ(0, dp[1]);
  }
  { // V CENTER
    grid.dataPosition = GridDataPosition::V_FACE_CENTER;
    Point2i cp = grid.cell(wp);
    Point2i dp = grid.dataCell(wp);
    EXPECT_EQ(cp[0], 1);
    EXPECT_EQ(cp[1], 1);
    EXPECT_EQ(0, dp[0]);
    EXPECT_EQ(1, dp[1]);
  }
}

TEST(ScalarGrid2f, Gradient) {
  float error = 1e-6;
  ScalarGrid2f grid;
  grid.dataPosition = GridDataPosition::CELL_CENTER;
  grid.set(10, 10, vec2(), vec2(1));
  grid.forEach([&](float &f, int i, int j) {
    UNUSED_VARIABLE(j);
    f = i;
  });
  grid.forEach([&](float &f, int i, int j) {
    UNUSED_VARIABLE(f);
    if (i == 0 || j == 0 || i == 9 || j == 9)
      return;
    vec2f g = grid.gradient(i, j);
    ASSERT_NEAR(g[0], 1.f, error);
    ASSERT_NEAR(g[1], 0.f, error);
  });
  ASSERT_NEAR(grid.gradient(2.5, 2.5f)[0], 1.f, error);
  ASSERT_NEAR(grid.gradient(2.5, 2.5f)[1], 0.f, error);
}

TEST(ScalarGrid2f, Sample) {
  float error = 1e-2;
  ScalarGrid2f grid;
  grid.set(10, 10, vec2(), vec2(1));
  grid.forEach([&](float &f, int i, int j) {
    UNUSED_VARIABLE(j);
    f = i;
  });
  grid.dataPosition = GridDataPosition::VERTEX_CENTER;
  for (float s = 1.f; s < 7; s += 0.1)
    ASSERT_NEAR(grid.sample(s, 5.f), s, error);
}
