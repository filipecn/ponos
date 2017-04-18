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
  // for (int i = 0; i < 2; i++)
  //  fastSweep2D<ponos::ZGrid<float>>(&grid, &grid, maxDist);
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << grid(i, j) << " ";
    std::cout << std::endl;
  }
  ponos::ZGrid<int> m(10, 10);
  ponos::ZGrid<float> phi(10, 10);
  phi.setAll(maxDist);
  ponos::ZGrid<float> v(10, 11);

  for (int x = 0; x < 10; x++) {
    for (int y = 0; y < 10; y++) {
      if (x < 2 || x > 7)
        phi(x, y) = -0.5f;
      if (y <= 5)
        v(x, y) = 1;
      if (y < 5)
        phi(x, y) = -0.5f;
      else
        m(x, y) = 1;
    }
  }
  std::cout << std::endl;
  for (int y = 9; y >= 0; y--) {
    for (int x = 0; x < 10; x++)
      std::cout << m(x, y) << " ";
    std::cout << std::endl;
  }

  std::cout << std::endl;
  for (int y = 10; y >= 0; y--) {
    for (int x = 0; x < 10; x++)
      std::cout << v(x, y) << " ";
    std::cout << std::endl;
  }

  for (int i = 0; i < 2; i++)
    ponos::fastSweep2D<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(&phi, &phi,
                                                                    &m, 0);
  std::cout << "distancias\n";
  std::cout << std::endl;
  for (int y = 9; y >= 0; y--) {
    for (int x = 0; x < 10; x++)
      std::cout << phi(x, y) << " ";
    std::cout << std::endl;
  }

  int nx = v.getDimensions()[0];
  int ny = v.getDimensions()[1];
  for (int i = 0; i < 4; i++) {
    ponos::sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(
        &v, &phi, &m, 1, 1, nx - 1, 1, ny - 1);
    ponos::sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(
        &v, &phi, &m, 1, 1, nx - 1, ny - 2, 0);
    ponos::sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(
        &v, &phi, &m, 1, nx - 2, 0, 1, ny - 1);
    ponos::sweep_y<ponos::ZGrid<float>, ponos::ZGrid<int>, int>(
        &v, &phi, &m, 1, nx - 2, 0, ny - 2, 0);
  }
  std::cout << std::endl;
  for (int y = 10; y >= 0; y--) {
    for (int x = 0; x < 10; x++)
      std::cout << v(x, y) << " ";
    std::cout << std::endl;
  }
}
