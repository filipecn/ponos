#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(Search, binary_search) {
  {
    std::vector<int> v;
    for (int i = 0; i < 10; i++)
      v.emplace_back(i);
    for (int i = 0; i < 10; i++)
      EXPECT_EQ(binary_search<int>(&v[0], v.size(), i), i);
  }
  {
    int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 5, 5};
    EXPECT_EQ(binary_search<int>(v, 10, 1), 0);
    EXPECT_EQ(binary_search<int>(v, 10, 2), 1);
    EXPECT_EQ(binary_search<int>(v, 10, 3), 2);
    EXPECT_EQ(binary_search<int>(v, 10, 4), 6);
    EXPECT_EQ(binary_search<int>(v, 10, 5), 8);
  }
  {
    int v[10] = {1, 2, 3, 3, 3, 3, 5, 5, 7, 7};
    EXPECT_EQ(binary_search<int>(v, 10, 6), -1);
    EXPECT_EQ(binary_search<int>(v, 10, 8), -1);
    EXPECT_EQ(binary_search<int>(v, 10, 0), -1);
  }
}

TEST(Search, upper_bound) {
  {
    int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 6, 6};
    EXPECT_EQ(upper_bound<int>(v, 10, 0), 0);
    EXPECT_EQ(upper_bound<int>(v, 10, 1), 1);
    EXPECT_EQ(upper_bound<int>(v, 10, 2), 2);
    EXPECT_EQ(upper_bound<int>(v, 10, 4), 8);
    EXPECT_EQ(upper_bound<int>(v, 10, 5), 8);
    EXPECT_EQ(upper_bound<int>(v, 10, 7), 10);
  }
}

TEST(Search, lower_bound) {
  {
    std::vector<int> v = {7, 15, 23, 24, 25, 26, 27, 28, 29, 30, 31, 39, 47, 55, 63};
    EXPECT_EQ(lower_bound<int>(&v[0], v.size(), 32), 10);
    EXPECT_EQ(lower_bound<int>(&v[2], v.size(), 32), 8);
  }
  {
    int v[10] = {1, 2, 3, 3, 3, 3, 4, 4, 6, 6};
    EXPECT_EQ(lower_bound<int>(v, 10, 6), 7);
    EXPECT_EQ(lower_bound<int>(v, 10, 6), 7);
    EXPECT_EQ(lower_bound<int>(v, 10, 5), 7);
    EXPECT_EQ(lower_bound<int>(v, 10, 4), 5);
    EXPECT_EQ(lower_bound<int>(v, 10, 3), 1);
    EXPECT_EQ(lower_bound<int>(v, 10, 2), 0);
    EXPECT_EQ(lower_bound<int>(v, 10, 1), -1);
  }
  {
    int v[8] = {0,1,2,3,4,5,6,7};
    EXPECT_EQ(lower_bound<int>(&v[6], 8 - 6, 7), 0);
    EXPECT_EQ(lower_bound<int>(&v[0], 8 - 0, 7), 6);
  }
}
