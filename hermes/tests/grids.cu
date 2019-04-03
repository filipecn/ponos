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