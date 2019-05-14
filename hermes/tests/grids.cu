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

TEST(RegularGrid, access) {
  { // PITCHED MEMORY
    vec3u res(128);
    RegularGrid3DPf grid(res);
    fill3(grid.data().accessor(), 1.f);
    // // RegularGrid3HLf h_grid(vec3u(8));
    // // memcpy(h_grid.data(), grid.data());
    // PitchedMemoryBlock3<MemoryLocation::DEVICE, float> pmb(res);
    // pmb.allocate();
    // LinearMemoryBlock3<MemoryLocation::HOST, float> lmb(res);
    // lmb.allocate();
    // memcpy(lmb, pmb);
    // memcpy(lmb, grid.data());
  }
}