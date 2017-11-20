#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestZGrid : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestZGrid);
  CPPUNIT_TEST(testInfNorm);
  CPPUNIT_TEST_SUITE_END();

  void testInfNorm();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestZGrid);

void TestZGrid::testInfNorm() {
  ZGrid<float> g(128, 128);
  ivec2 ij;
  ivec2 D = g.getDimensions();
  float m = 0.f;
  HaltonSequence rng(3);
  FOR_INDICES0_2D(D, ij) {
    g(ij) = rng.randomFloat() - 0.5f;
    m = std::max(m, std::fabs(g(ij)));
  }
  CPPUNIT_ASSERT(IS_EQUAL(m, g.infNorm()));
}
