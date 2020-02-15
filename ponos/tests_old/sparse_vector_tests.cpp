#include <gtest/gtest.h>
#include <ponos/ponos.h>

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

TEST(SparseVector, Swap) {
  SparseVector<double> evens(1000, 500);
  for (size_t i = 0; i < 1000; i += 2)
    evens.insert(i, i);
  evens.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i), 1e-8);
    EXPECT_EQ((i % 2), 0u);
  });
  SparseVector<double> odds(1000, 500);
  for (size_t i = 1; i < 1000; i += 2)
    odds.insert(i, i);
  odds.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i), 1e-8);
    EXPECT_EQ((i % 2), 1u);
  });
  evens.swap(&odds);
  odds.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i), 1e-8);
    EXPECT_EQ((i % 2), 0u);
  });
  evens.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i), 1e-8);
    EXPECT_EQ((i % 2), 1u);
  });
}

TEST(SparseVector, ScalarMultiplication) {
  SparseVector<double> v(1000, 500);
  for (size_t i = 1; i < 1000; i += 2)
    v.insert(i, i);
  v *= 3.0;
  EXPECT_EQ(v.elementCount(), 500u);
  v.iterate([](double &v, size_t i) {
    ASSERT_NEAR(v, static_cast<double>(i * 3), 1e-8);
    EXPECT_EQ((i % 2), 1u);
  });
}
