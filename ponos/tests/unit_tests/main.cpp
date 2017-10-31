#include <ponos.h>
#include <gtest/gtest.h>
#include <fstream>

using namespace ponos;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  return ret;
}
