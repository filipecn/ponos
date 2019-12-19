#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(FDMatrix2, acc) {
  // y
  // |
  //  ---x
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  vec2u size(4);
  FDMatrix2H h_A(size);
  EXPECT_EQ(h_A.gridSize(), size);
  EXPECT_EQ(h_A.size(), 4u * 4u);
  auto iAcc = h_A.indexDataAccessor();
  int curIndex = 0;
  for (size_t j = 0; j < size.y; j++)
    for (size_t i = 0; i < size.x; i++)
      if (j == 0 || j == size.y - 1 || i == 0 || i == size.x - 1)
        iAcc(i, j) = -1;
      else
        iAcc(i, j) = curIndex++;
  // std::cerr << h_A.indexData() << std::endl;
  auto acc = h_A.accessor();
  for (size_t j = 1; j < size.y - 1; j++)
    for (size_t i = 1; i < size.x - 1; i++) {
      if (i > 1)
        EXPECT_NE(acc.elementIndex(i - 1, j), -1);
      else if (i == 1)
        EXPECT_EQ(acc.elementIndex(i - 1, j), -1);
      if (j > 1)
        EXPECT_NE(acc.elementIndex(i, j - 1), -1);
      else if (j == 1)
        EXPECT_EQ(acc.elementIndex(i, j - 1), -1);
      if (i < size.x - 2)
        EXPECT_NE(acc.elementIndex(i + 1, j), -1);
      else if (i == size.x - 2)
        EXPECT_EQ(acc.elementIndex(i + 1, j), -1);
      if (j < size.y - 2)
        EXPECT_NE(acc.elementIndex(i, j + 1), -1);
      else if (j == size.y - 2)
        EXPECT_EQ(acc.elementIndex(i, j + 1), -1);
      acc(i, j, i, j) = 6;
      acc(i, j, i + 1, j) = 1;
      acc(i, j, i, j + 1) = 2;
    }

  std::cerr << h_A << std::endl;
  std::cerr << acc << std::endl;
}

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
  // std::cerr << h_A.indexData() << std::endl;
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

  // std::cerr << h_A << std::endl;
  // std::cerr << acc << std::endl;
}

TEST(FDMatrix2, mul) {
  // 3 4 5 0  0     14
  // 4 3 0 5  1  =  18
  // 5 0 3 4  2     18
  // 0 5 4 3  3     22
  vec2u size(2);
  FDMatrix2H h_A(size);
  auto haAcc = h_A.accessor();
  auto iAcc = h_A.indexDataAccessor();
  int index = 0;
  for (int j = 0; j < size.y; j++)
    for (int i = 0; i < size.x; i++)
      iAcc(i, j) = index++;
  // std::cerr << h_A.indexData() << std::endl;
  for (int j = 0; j < size.y; j++)
    for (int i = 0; i < size.x; i++) {
      haAcc(i, j, i, j) = 3;
      haAcc(i, j, i + 1, j) = 4;
      haAcc(i, j, i, j + 1) = 5;
    }
  // std::cerr << haAcc << std::endl;
  MemoryBlock1Hf h_x(h_A.size(), 0);
  auto hxAcc = h_x.accessor();
  for (int i = 0; i < h_x.size(); i++)
    hxAcc[i] = i;
  int idx = 0;
  float ans[4] = {14, 18, 18, 22};
  for (int j = 0; j < size.y; j++)
    for (int i = 0; i < size.x; i++) {
      EXPECT_EQ(ans[idx],
                (iAcc.isIndexValid(i - 1, j) ? iAcc(i - 1, j) : 0) *
                        haAcc(i, j, i - 1, j) +
                    (iAcc.isIndexValid(i, j - 1) ? iAcc(i, j - 1) : 0) *
                        haAcc(i, j, i, j - 1) +
                    (iAcc.isIndexValid(i, j) ? iAcc(i, j) : 0) *
                        haAcc(i, j, i, j) +
                    (iAcc.isIndexValid(i + 1, j) ? iAcc(i + 1, j) : 0) *
                        haAcc(i, j, i + 1, j) +
                    (iAcc.isIndexValid(i, j + 1) ? iAcc(i, j + 1) : 0) *
                        haAcc(i, j, i, j + 1));
      idx++;
    }
  // std::cerr << haAcc << std::endl;
  MemoryBlock1Df d_x(h_A.size(), 1.f), d_b(h_A.size(), 0.f);
  memcpy(d_x, h_x);
  FDMatrix2D d_A(size);
  d_A.copy(h_A);
  mul(d_A, d_x, d_b);
  MemoryBlock1Hf h_b(4, 0.f);
  memcpy(h_b, d_b);
  auto acc = h_b.accessor();
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(acc[i], ans[i]);
}

TEST(FDMatrix3, mul) {
  // 4 1 2 0 3 0 0 0  0  17
  // 1 4 0 2 0 3 0 0  1  25
  // 2 0 4 1 0 0 3 0  2  29
  // 0 2 1 4 0 0 0 3  3  37
  // 3 0 0 0 4 1 2 0  4  33
  // 0 3 0 0 1 4 0 2  5  41
  // 0 0 3 0 2 0 4 1  6  45
  // 0 0 0 3 0 2 1 4  7  53
  vec3u size(2);
  FDMatrix3H h_A(size);
  auto haAcc = h_A.accessor();
  auto iAcc = h_A.indexDataAccessor();
  int index = 0;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        iAcc(i, j, k) = index++;
  // std::cerr << h_A.indexData() << std::endl;
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
  MemoryBlock1Hf h_x(h_A.size(), 0);
  auto hxAcc = h_x.accessor();
  for (int i = 0; i < h_x.size(); i++)
    hxAcc[i] = i;
  int idx = 0;
  float ans[8] = {17, 25, 29, 37, 33, 41, 45, 53};
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++) {
        EXPECT_EQ(ans[idx],
                  (iAcc.isIndexValid(i - 1, j, k) ? iAcc(i - 1, j, k) : 0) *
                          haAcc(i, j, k, i - 1, j, k) +
                      (iAcc.isIndexValid(i, j - 1, k) ? iAcc(i, j - 1, k) : 0) *
                          haAcc(i, j, k, i, j - 1, k) +
                      (iAcc.isIndexValid(i, j, k - 1) ? iAcc(i, j, k - 1) : 0) *
                          haAcc(i, j, k, i, j, k - 1) +
                      (iAcc.isIndexValid(i, j, k) ? iAcc(i, j, k) : 0) *
                          haAcc(i, j, k, i, j, k) +
                      (iAcc.isIndexValid(i + 1, j, k) ? iAcc(i + 1, j, k) : 0) *
                          haAcc(i, j, k, i + 1, j, k) +
                      (iAcc.isIndexValid(i, j + 1, k) ? iAcc(i, j + 1, k) : 0) *
                          haAcc(i, j, k, i, j + 1, k) +
                      (iAcc.isIndexValid(i, j, k + 1) ? iAcc(i, j, k + 1) : 0) *
                          haAcc(i, j, k, i, j, k + 1));
        idx++;
      }
  // std::cerr << haAcc << std::endl;
  MemoryBlock1Df d_x(h_A.size(), 1.f), d_b(h_A.size(), 0.f);
  memcpy(d_x, h_x);
  // std::cerr << d_x << std::endl;
  FDMatrix3D d_A(size);
  d_A.copy(h_A);
  mul(d_A, d_x, d_b);
  // std::cerr << d_b << std::endl;
  MemoryBlock1Hf h_b(8, 0.f);
  memcpy(h_b, d_b);
  auto acc = h_b.accessor();
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(acc[i], ans[i]);
}

TEST(PCG, fdMatrix) {
  return;
  // 1 0 0 0 0 0 0 0       x       1
  // 0 4 0 2 0 3 0 0       x       2
  // 0 0 4 1 0 0 3 0       x       3
  // 0 2 1 4 0 0 0 3   x   x   =   4
  // 0 0 0 0 4 1 2 0       x       5
  // 0 3 0 0 1 4 0 2       x       6
  // 0 0 3 0 2 0 4 1       x       7
  // 0 0 0 3 0 2 1 4       x       8
  vec3u size(2);
  FDMatrix3H h_A(size);
  auto haAcc = h_A.accessor();
  auto iAcc = h_A.indexDataAccessor();
  int index = 0;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        iAcc(i, j, k) = index++;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++) {
        haAcc(i, j, k, i, j, k) = 6;
        haAcc(i, j, k, i + 1, j, k) = 1;
        haAcc(i, j, k, i, j + 1, k) = 1;
        haAcc(i, j, k, i, j, k + 1) = 1;
      }
  MemoryBlock1Hd h_b(h_A.size(), 0);
  auto acc = h_b.accessor();
  for (int i = 0; i < h_b.size(); i++)
    acc[i] = 1; // i + 1;
  MemoryBlock1Dd d_x(h_A.size(), 0), d_b(h_A.size(), 0);
  memcpy(d_b, h_b);
  FDMatrix3D d_A(size);
  d_A.copy(h_A);
  pcg(d_x, d_A, d_b, 100, 1e-4);
  std::cerr << d_x << std::endl;
}