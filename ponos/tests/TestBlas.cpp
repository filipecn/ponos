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
  { // TEST FDM CG2f
    size_t n = 7, m = 7;
    FDMLinearSystem2f system;
    system.resize(n, m);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < m; j++) {
        system.A(i, j).center = 19600;
        if (i + 1 < 7)
          system.A(i, j).right = -4900;
        else
          system.A(i, j).right = 0;
        if (j + 1 < 7)
          system.A(i, j).up = -4900;
        else
          system.A(i, j).up = 0;
      }
    system.x.set(0.0);
    system.b.set(1.0);
    FDMCGSolver2f<NullCGPreconditioner<FDMBlas2f>> solver(100, 1e-7);
    CPPUNIT_ASSERT(solver.solve(&system));
    FDMVector2Df B(n, m);
    FDMBlas2f::mvm(system.A, system.x, &B);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < m; j++)
        CPPUNIT_ASSERT(std::fabs(B(i, j) - system.b(i, j)) < 1e-5);
    FDMJacobiSolver2f jsolver(100, 1e-5, 4);
    CPPUNIT_ASSERT(jsolver.solve(&system));
    B.set(0.f);
    FDMBlas2f::mvm(system.A, system.x, &B);
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < m; j++)
        CPPUNIT_ASSERT(std::fabs(B(i, j) - system.b(i, j)) < 1e-5);
  }
  // CPPUNIT_ASSERT(IS_EQUAL(v(i, j), 1.f));
}
