#include <poseidon.h>
#include <gtest/gtest.h>

using namespace ponos;
using namespace poseidon;

TEST(GridSolver2D, Build) {
  GridSolver2D solver;
  solver.set(10, BBox2D(Point2(), Point2(1, 1)));
  StaggeredGrid2f &grid = solver.getGrid();
  EXPECT_EQ(grid.p.width, 10u);
  EXPECT_EQ(grid.p.height, 10u);
  EXPECT_EQ(grid.u.width, 11u);
  EXPECT_EQ(grid.u.height, 10u);
  EXPECT_EQ(grid.v.width, 10u);
  EXPECT_EQ(grid.v.height, 11u);
}
TEST(GridSolver2D, EnforceBoundaries) {
  // float error = 1e-8;
  GridSolver2D solver;
  solver.set(10, BBox2D(Point2(), Point2(1, 1)));
  StaggeredGrid2f &grid = solver.getGrid();
  grid.u.setAll(1.f);
  grid.v.setAll(1.f);
  SimulationScene2D &scene = solver.getScene();
  Collider2D c;
  c.implicitCurve.reset(new ImplicitCircle(Point2(0.5f), 0.25));
  scene.addCollider(c);
  std::cout << scene.getSDF() << std::endl;
  std::cout << &grid.u << std::endl;
  solver.markCells();
  solver.enforceBoundaries();
  std::cout << &grid.u << std::endl;
}
