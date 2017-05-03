#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestBlas : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestBlas);
  CPPUNIT_TEST(testPCG);
  CPPUNIT_TEST_SUITE_END();

  void testPCG();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBlas);

void TestBlas::testPCG() {
  // CPPUNIT_ASSERT(IS_EQUAL(v(i, j), 1.f));
}
