//
// Created by filipecn on 2/26/18.
//

#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

bool isIdentity(const Matrix4x4 &m) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (i != j && !IS_EQUAL(m.m[i][j], 0.f))
        return false;
      else if (i == j && !IS_EQUAL(m.m[i][j], 1.f))
        return false;
  return true;
}

TEST(Matrix4x4, Constructors) {
  { // default
    { // make identity
      Matrix4x4 m;
      EXPECT_TRUE(isIdentity(m));
    }
    {
      Matrix4x4 m(false);
      for (auto &row : m.m)
        for (float v : row)
          EXPECT_FLOAT_EQ(v, 0.f);
    }
  }
  { // initializer list constructors
    { // row major
      Matrix4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, false);
      for (auto &row : m.m)
        for (float v : row) {
          static float k = 1;
          EXPECT_FLOAT_EQ(v, k);
          k += 1.f;
        }
    }
    { // column major
      Matrix4x4 m({1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16}, true);
      for (auto &row : m.m)
        for (float v : row) {
          static float k = 1;
          EXPECT_FLOAT_EQ(v, k);
          k += 1.f;
        }
    }
  }
  { // from float[]
    { // row major
      float a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
      Matrix4x4 m(a, false);
      for (auto &row : m.m)
        for (float v : row) {
          static float k = 1;
          EXPECT_FLOAT_EQ(v, k);
          k += 1.f;
        }
    }
    { // column major
      float a[16] = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};
      Matrix4x4 m(a, true);
      for (auto &row : m.m)
        for (float v : row) {
          static float k = 1;
          EXPECT_FLOAT_EQ(v, k);
          k += 1.f;
        }
    }
  }
}

TEST(Matrix4x4, Inverse) {
  Matrix4x4 m(true);
  EXPECT_TRUE(isIdentity(m));
  Matrix4x4 mm = m * inverse(m);
  EXPECT_TRUE(isIdentity(mm));
  m.m[0][0] = 0;
  m.m[1][0] = 0;
  m.m[2][0] = -1;
  m.m[3][0] = 0;

  m.m[0][1] = 0;
  m.m[1][1] = 1;
  m.m[2][1] = 0;
  m.m[3][1] = 0;

  m.m[0][2] = 1;
  m.m[1][2] = 0;
  m.m[2][2] = 0;
  m.m[3][2] = -1;

  m.m[0][3] = 0;
  m.m[1][3] = 0;
  m.m[2][3] = 0;
  m.m[3][3] = 1;

  mm = m * inverse(m);
  EXPECT_TRUE(isIdentity(mm));
}

TEST(Matrix, isIdentity) {
  // TODO
}

TEST(MatrixNM, Constructors) {
  { // COLUMN AND ROW MAJOR CONSTRUCTORS
    Matrix<float, 3, 3> cm({1, 2, 3, 4, 5, 6, 7, 8, 9}, true);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        static float k = 1;
        EXPECT_FLOAT_EQ(cm.m[j][i], k);
        k += 1.f;
      }
    Matrix<float, 3, 3> rm({1, 2, 3, 4, 5, 6, 7, 8, 9}, false);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        static float k = 1;
        EXPECT_FLOAT_EQ(cm.m[i][j], k);
        k += 1.f;
      }
  }
}

TEST(MatrixNM, arithmetic) {
  { // MATRIX x VECTOR
    Matrix<float, 3, 3> m({1, 2, 3, 10, 20, 30, 100, 200, 300});
    Vector<float, 3> v({3, 2, 1});
    Vector<float, 3> a = m * v;
    EXPECT_FLOAT_EQ(a[0], static_cast<float>(1 * 3 + 2 * 2 + 3 * 1));
    EXPECT_FLOAT_EQ(a[1], static_cast<float>(10 * 3 + 20 * 2 + 30 * 1));
    EXPECT_FLOAT_EQ(a[2], static_cast<float>(100 * 3 + 200 * 2 + 300 * 1));
  }
  { // MATRIX x MATRIX
    //Matrix<float, 3, 2> A({1, 1, 1, 1, 1, 1});
    //Matrix<float, 2, 3> B({2, 2, 2, 2, 2, 2});
    //Matrix<float, 3, 3> C = A * B;
    // TODO
  }
}

TEST(MatrixNM, inverse) {
  { // TEST x INVERSE
    Matrix<float, 5, 5> m({12, 13, 8, 4, 9, 5, 11, 10, 14, 15, 2, 3, 16, 6, 7,
                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
    Matrix<float, 5, 5> mi = inverse(m);
    float M[5][5] = {{-0.0135f, -0.2108f, -0.0464f, 0.9679f, -0.6429f},
                     {0.0667f, 0.2534f, 0.0145f, -2.2275f, 1.6259f},
                     {0.0048f, 0.0039f, 0.088f, -0.46f, 0.3439f},
                     {-0.1557f, 0.0754f, -0.0338f, -1.4692f, 1.2062f},
                     {0.0977f, -0.1219f, -0.0222f, 2.9888f, -2.3331f}};
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        EXPECT_FLOAT_EQ(mi.m[i][j], M[i][j]);
  }
}