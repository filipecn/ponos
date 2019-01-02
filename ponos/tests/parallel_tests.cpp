//
// Created by filipecn on 2018-12-26.
//

#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(Parallel, parallel_for) {
  {
    parallel_for([](int i) {
      }, 1000, 100);
  }
}