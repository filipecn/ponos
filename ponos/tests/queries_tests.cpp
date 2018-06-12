// Created by filipecn on 3/17/18.
#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(Queries, ray_segment) {
  Segment3 s(Point3(-15, 0, 0), Point3(10, 0, 0));
  Ray3 r(Point3(0, 20, 0), vec3(0, -1, 0));
  EXPECT_TRUE(ray_segment_intersection(r, s));
  Ray3 r2(Point3(15, 0, 0), vec3(0, -1, 0));
  EXPECT_FALSE(ray_segment_intersection(r2, s));

  Ray3 r3(Point3(19.266, 1.79213, 5.06042),
          vec3(-0.896717, -0.296824, -0.32832));
  //std::cout << r3(11.1965);
  Ray3 sr(s.a, s.b - s.a);
  //std::cout << sr(0.969035);
  EXPECT_FALSE(ray_segment_intersection(r3, s));

  Ray3 r4(Point3(10.5076, 1.30321, 1.44688),
          vec3(-0.878212, -0.321182, -0.354382));
  EXPECT_TRUE(ray_segment_intersection(r4, s));
}

TEST(Queries, closestPointTriangle) {
  {
    auto cp = closest_point_triangle(Point3(0,0,0), Point3(-1, -1, 0),
                                     Point3(1, -1, 0), Point3(0,1,0));
    EXPECT_EQ(cp, Point3(0,0,0));
    cp = closest_point_triangle(Point3(-2,-2,0), Point3(-1, -1, 0),
                                     Point3(1, -1, 0), Point3(0,1,0));
    EXPECT_EQ(cp, Point3(-1,-1,0));
    cp = closest_point_triangle(Point3(0,0,-2), Point3(-1, -1, 0),
                                Point3(1, -1, 0), Point3(0,1,0));
    EXPECT_EQ(cp, Point3(0,0,0));
    cp = closest_point_triangle(Point3(0,-2,0), Point3(-1, -1, 0),
                                Point3(1, -1, 0), Point3(0,1,0));
    EXPECT_EQ(cp, Point3(0,-1,0));
  }
}
