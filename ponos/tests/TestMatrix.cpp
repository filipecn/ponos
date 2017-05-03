#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestMatrix : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestMatrix);
  CPPUNIT_TEST(testInverse);
  CPPUNIT_TEST(testIdentity);
  CPPUNIT_TEST(testMatrixNM);
  CPPUNIT_TEST_SUITE_END();

  void testInverse();
  void testIdentity();
  void testMatrixNM();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMatrix);

void print(const Matrix4x4 &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      std::cout << m.m[i][j] << " ";
    std::cout << std::endl;
  }
}

bool isIdentity(const Matrix4x4 &m) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (i != j && !IS_EQUAL(m.m[i][j], 0.f))
        return false;
      else if (i == j && !IS_EQUAL(m.m[i][j], 1.f))
        return false;
  return true;
}

void TestMatrix::testInverse() {
  Matrix4x4 m;
  CPPUNIT_ASSERT(isIdentity(m));
  Matrix4x4 mm = m * inverse(m);
  CPPUNIT_ASSERT(isIdentity(mm));
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
  // print(m);
  // print(inverse(m));
  mm = m * inverse(m);
  CPPUNIT_ASSERT(isIdentity(mm));
}

void TestMatrix::testIdentity() { CPPUNIT_ASSERT(true); }

void TestMatrix::testMatrixNM() {
  { // COLUMN AND ROW MAJOR CONSTRUCTORS
    Matrix<float, 3, 3> cm({1, 2, 3, 4, 5, 6, 7, 8, 9}, true);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        static int k = 1;
        CPPUNIT_ASSERT(cm.m[j][i] == k++);
      }
    Matrix<float, 3, 3> rm({1, 2, 3, 4, 5, 6, 7, 8, 9}, false);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        static int k = 1;
        CPPUNIT_ASSERT(rm.m[i][j] == k++);
      }
  }
  { // MATRIX x VECTOR
    Matrix<float, 3, 3> m({1, 2, 3, 10, 20, 30, 100, 200, 300});
    Vector<float, 3> v({3, 2, 1});
    Vector<float, 3> a = m * v;
    CPPUNIT_ASSERT(a[0] == 1 * 3 + 2 * 2 + 3 * 1);
    CPPUNIT_ASSERT(a[1] == 10 * 3 + 20 * 2 + 30 * 1);
    CPPUNIT_ASSERT(a[2] == 100 * 3 + 200 * 2 + 300 * 1);
  }
  { // MATRIX x MATRIX
    Matrix<float, 3, 2> A({1, 1, 1, 1, 1, 1});
    Matrix<float, 2, 3> B({2, 2, 2, 2, 2, 2});
    Matrix<float, 3, 3> C = A * B;
  }
  { // TEST x INVERSE
    Matrix<float, 4, 4> m({3, 4, 2, 7, 2, 6, 7, 3, 2, 4, 1, 2, 7, 8, 4, 9});
    inverse(m);
  }
}
