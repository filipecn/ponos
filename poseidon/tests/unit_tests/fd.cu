#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(FDMatrix3, acc) {
  // y
  // |
  //  ---x
  // z = 0
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  // z = 1
  //  S S S S  - - - -
  //  S F F S  - 2 3 -
  //  S F F S  - 0 1 -
  //  S S S S  - - - -
  // z = 2
  //  S S S S  - - - -
  //  S F F S  - 6 7 -
  //  S F F S  - 4 5 -
  //  S S S S  - - - -
  // z = 3
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  vec3u size(4);
  FDMatrix3H h_A(size);
  EXPECT_EQ(h_A.gridSize(), size);
  EXPECT_EQ(h_A.size(), 4u * 4u * 4u);
  auto iAcc = h_A.indexDataAccessor();
  int curIndex = 0;
  for (size_t k = 0; k < size.z; k++)
    for (size_t j = 0; j < size.y; j++)
      for (size_t i = 0; i < size.x; i++)
        if (k == 0 || k == size.z - 1 || j == 0 || j == size.y - 1 || i == 0 ||
            i == size.x - 1)
          iAcc(i, j, k) = -1;
        else
          iAcc(i, j, k) = curIndex++;
  std::cerr << h_A.indexData() << std::endl;
  auto acc = h_A.accessor();
  for (size_t k = 1; k < size.z - 1; k++)
    for (size_t j = 1; j < size.y - 1; j++)
      for (size_t i = 1; i < size.x - 1; i++) {
        if (i > 1)
          EXPECT_NE(acc.elementIndex(i - 1, j, k), -1);
        else if (i == 1)
          EXPECT_EQ(acc.elementIndex(i - 1, j, k), -1);
        if (j > 1)
          EXPECT_NE(acc.elementIndex(i, j - 1, k), -1);
        else if (j == 1)
          EXPECT_EQ(acc.elementIndex(i, j - 1, k), -1);
        if (k > 1)
          EXPECT_NE(acc.elementIndex(i, j, k - 1), -1);
        else if (k == 1)
          EXPECT_EQ(acc.elementIndex(i, j, k - 1), -1);
        if (i < size.x - 2)
          EXPECT_NE(acc.elementIndex(i + 1, j, k), -1);
        else if (i == size.x - 2)
          EXPECT_EQ(acc.elementIndex(i + 1, j, k), -1);
        if (j < size.y - 2)
          EXPECT_NE(acc.elementIndex(i, j + 1, k), -1);
        else if (j == size.y - 2)
          EXPECT_EQ(acc.elementIndex(i, j + 1, k), -1);
        if (k < size.z - 2)
          EXPECT_NE(acc.elementIndex(i, j, k + 1), -1);
        else if (k == size.z - 2)
          EXPECT_EQ(acc.elementIndex(i, j, k + 1), -1);
        acc(i, j, k, i, j, k) = 6;
        acc(i, j, k, i + 1, j, k) = 1;
        acc(i, j, k, i, j + 1, k) = 2;
        acc(i, j, k, i, j, k + 1) = 3;
      }

  std::cerr << h_A << std::endl;
  std::cerr << acc << std::endl;
}

TEST(FDMatrix3, mul) {
  // 4 1 2 0 3 0 0 0       1
  // 1 4 0 2 0 3 0 0       1
  // 2 0 4 1 0 0 3 0       1
  // 0 2 1 4 0 0 0 3   x   1
  // 3 0 0 0 4 1 2 0       1
  // 0 3 0 0 1 4 0 2       1
  // 0 0 3 0 2 0 4 1       1
  // 0 0 0 3 0 2 1 4       1
  vec3u size(2);
  FDMatrix3H h_A(size);
  auto haAcc = h_A.accessor();
  auto iAcc = h_A.indexDataAccessor();
  int index = 0;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        iAcc(i, j, k) = index++;
  std::cerr << h_A.indexData() << std::endl;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++) {
        haAcc(i, j, k, i - 1, j, k) = 1;
        haAcc(i, j, k, i, j - 1, k) = 2;
        haAcc(i, j, k, i, j, k - 1) = 3;
        haAcc(i, j, k, i, j, k) = 4;
        haAcc(i, j, k, i + 1, j, k) = 5;
        haAcc(i, j, k, i, j + 1, k) = 6;
        haAcc(i, j, k, i, j, k + 1) = 7;
      }
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        EXPECT_EQ(
            10, haAcc(i, j, k, i - 1, j, k) + haAcc(i, j, k, i, j - 1, k) +
                    haAcc(i, j, k, i, j, k - 1) + haAcc(i, j, k, i, j, k) +
                    haAcc(i, j, k, i + 1, j, k) + haAcc(i, j, k, i, j + 1, k) +
                    haAcc(i, j, k, i, j, k + 1));
  std::cerr << haAcc << std::endl;
  MemoryBlock1Df d_x(h_A.size()), d_b(h_A.size());
  d_x.allocate();
  d_b.allocate();
  fill1(d_x.accessor(), 1.f);
  FDMatrix3D d_A(size);
  d_A.copy(h_A);
  mul(d_A, d_x, d_b);
  std::cerr << d_b << std::endl;
}