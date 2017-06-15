#include <ponos.h>
#include <gtest/gtest.h>

using namespace ponos;

// 0 1 0 3 0 5 0 7 ...
TEST(SparseVector, Iterate) {
  SparseVector<double> v(1000, 500);
  for (size_t i = 1; i < 1000; i += 2)
    v.insert(i, i);
  EXPECT_EQ(v.elementCount(), 500u);
  v.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i), 1e-8);
    EXPECT_EQ((i % 2), 1u);
  });
}
