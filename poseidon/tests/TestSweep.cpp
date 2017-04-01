#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

using namespace poseidon;

class TestSweep : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestSweep);
  CPPUNIT_TEST(testDistance);
  CPPUNIT_TEST_SUITE_END();

  void testDistance();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestSweep);

void TestSweep::testDistance() {
  ponos::ZGrid<float> grid(10, 10);
  float maxDist = 10 * 3;
  grid.setAll(maxDist);
  grid(4, 4) = 0.5f;
  grid(5, 4) = 0.5f;
  grid(5, 5) = 0.5f;
  grid(4, 5) = 0.5f;
  for (int i = 0; i < 2; i++)
    fastSweep2D<ponos::ZGrid<float>>(&grid, &grid, maxDist);
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << grid(i, j) << " ";
    std::cout << std::endl;
  }
  ponos::ZGrid<int> m(10, 10);
  ponos::ZGrid<float> v(10, 10);
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      if (i < 5)
        m(i, j) = 1;
      else
        v(i, j) = 1;
  }
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << m(i, j) << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << v(i, j) << " ";
    std::cout << std::endl;
  }

  int nx = grid.getDimensions()[0];
  int ny = grid.getDimensions()[1];
  for (int i = 0; i < 4; i++) {
    sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(&v, &grid, &m, 1, 1,
                                                         nx - 1, 1, ny - 1);
    sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(&v, &grid, &m, 1, 1,
                                                         nx - 1, ny - 2, 0);
    sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(&v, &grid, &m, 1,
                                                         nx - 2, 0, 1, ny - 1);
    sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(&v, &grid, &m, 1,
                                                         nx - 2, 0, ny - 2, 0);
  }
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << v(i, j) << " ";
    std::cout << std::endl;
  }
}
