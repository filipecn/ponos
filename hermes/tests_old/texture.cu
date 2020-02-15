#include "utils.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;

TEST(Texture3, Constructor) {
  unsigned int width = 8;
  unsigned int height = 8;
  unsigned int depth = 8;
  Texture3<float> t(width, height, depth);
  fill<float>(t, 1.f);
  std::cerr << t << std::endl;
}

TEST(Texture, fill) {
  unsigned int width = 16;
  unsigned int height = 16;
  Texture<float> t(width, height);
  fillTexture(t);
  std::cerr << t << std::endl;
}