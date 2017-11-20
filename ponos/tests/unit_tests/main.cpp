#include <fstream>
#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  return ret;
}
