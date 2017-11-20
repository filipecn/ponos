#include <ponos.h>
#include <cppunit/extensions/HelperMacros.h>
#include <iostream>

using namespace ponos;

class TestGrid : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestGrid);
  CPPUNIT_TEST(testScalarGrid);
  CPPUNIT_TEST_SUITE_END();

  void testScalarGrid();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGrid);

void TestGrid::testScalarGrid() {
  { // TEST ACCESS
    ScalarGrid2f grid(10, 10);
    grid.setAll(1.f);
    ivec2 ij;
    FOR_INDICES0_2D(ij, grid.getDimensions())
    CPPUNIT_ASSERT(IS_EQUAL(grid(ij), 1.f));
  }
}
