#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(RBF2D, Constructor) {
  std::vector<Point2> points(100);
  for (auto &p : points)
    p = Point2(1, 1);
  // std::vector<float> f(100, 1.f);
  // RBF2D<float, Point2> rbd(points, f, new PowKernel<float>(5));
}
