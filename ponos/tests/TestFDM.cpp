#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestFDM : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestFDM);
  CPPUNIT_TEST(testFDMBlas);
  CPPUNIT_TEST_SUITE_END();

  void testFDMBlas();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFDM);

void TestFDM::testFDMBlas() {
  { // set scalar to vector
    FDMVector2Df v(10, 10);
    FDMBlas2f::set(1.f, &v);
    for (size_t i = 0; i < 10; i++)
      for (size_t j = 0; j < 10; j++)
        CPPUNIT_ASSERT(IS_EQUAL(v(i, j), 1.f));
  }
  { // set vector to vector
    FDMVector2Df a(10, 10);
    FDMVector2Df b(10, 10);
    FDMBlas2f::set(1.f, &a);
    FDMBlas2f::set(3.f, &b);
    for (size_t i = 0; i < 10; i++)
      for (size_t j = 0; j < 10; j++)
        CPPUNIT_ASSERT(IS_EQUAL(a(i, j) * 3.f, b(i, j)));
  }
  // FDMMatrix2Df m(10, 10);
}
