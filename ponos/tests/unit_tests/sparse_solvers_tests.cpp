#include <ponos.h>
#include <gtest/gtest.h>

using namespace ponos;

TEST(SparseSolvers, j_gs_sor) {
  SparseLinearSystemd system;
  system.A.set(8, 8, 8 * 6);
  system.A.insert(0, 0, 3.1);
  system.A.insert(0, 1, -1.0);
  system.A.insert(0, 2, -1.0);
  system.A.insert(0, 5, -1.0);
  system.A.insert(1, 0, -1.0);
  system.A.insert(1, 1, 4.0);
  system.A.insert(1, 2, -1.0);
  system.A.insert(1, 4, -1.0);
  system.A.insert(1, 6, -1.0);
  system.A.insert(2, 0, -1.0);
  system.A.insert(2, 1, -1.0);
  system.A.insert(2, 2, 5.0);
  system.A.insert(2, 3, -1.0);
  system.A.insert(2, 5, -1.0);
  system.A.insert(2, 7, -1.0);
  system.A.insert(3, 2, -1.0);
  system.A.insert(3, 3, 4.1);
  system.A.insert(3, 4, -1.0);
  system.A.insert(3, 5, -1.0);
  system.A.insert(3, 6, -1.0);
  system.A.insert(4, 1, -1.0);
  system.A.insert(4, 3, -1.0);
  system.A.insert(4, 4, 4.5);
  system.A.insert(4, 5, -1.0);
  system.A.insert(4, 7, -1.0);
  system.A.insert(5, 0, -1.0);
  system.A.insert(5, 2, -1.0);
  system.A.insert(5, 3, -1.0);
  system.A.insert(5, 4, -1.0);
  system.A.insert(5, 5, 5.0);
  system.A.insert(5, 6, -1.0);
  system.A.insert(6, 1, -1.0);
  system.A.insert(6, 3, -1.0);
  system.A.insert(6, 5, -1.0);
  system.A.insert(6, 6, 3.5);
  system.A.insert(6, 7, -1.0);
  system.A.insert(7, 2, -1.0);
  system.A.insert(7, 4, -1.0);
  system.A.insert(7, 6, -1.0);
  system.A.insert(7, 7, 3.3);
  system.b.set(8, 5);
  system.b.insert(0, 0.1);
  system.b.insert(3, 0.1);
  system.b.insert(4, 0.5);
  system.b.insert(6, -0.5);
  system.b.insert(7, 0.3);
  {
    system.x.set(8, 8);
    for (size_t i = 0; i < 8; i++)
      system.x.insert(i, -10.0);
    SparseJacobiSolverd solver(1000, 1e-4, 4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
  {
    system.x.set(8, 8);
    for (size_t i = 0; i < 8; i++)
      system.x.insert(i, -10.0);
    SparseGaussSeidelSolverd solver(1000, 1e-4, 4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
  {
    system.x.set(8, 8);
    for (size_t i = 0; i < 8; i++)
      system.x.insert(i, -10.0);
    SparseSORSolverd solver(1.79, 1000, 1e-4, 4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
}

TEST(SparseSolvers, CG) {
  SparseLinearSystemd system;
  system.A.set(2, 2, 2 * 2);
  system.A.insert(0, 0, 4);
  system.A.insert(0, 1, 1);
  system.A.insert(1, 0, 1);
  system.A.insert(1, 1, 3);
  system.b.set(2, 2);
  system.b.insert(0, 1);
  system.b.insert(1, 2);
  {
    system.x.set(2, 2);
    system.x.insert(0, 2);
    system.x.insert(1, 1);
    SparseCGSolverd<NullCGPreconditioner<SparseBlas2d>> solver(2, 1e-4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
}

TEST(SparseSolvers, PCG) {
  SparseLinearSystemd system;
  system.A.set(8, 8, 4 + 5 + 6 + 5 + 5 + 6 + 5 + 4);
  system.A.insert(0, 0, 3.100);
  system.A.insert(0, 1, -1.000);
  system.A.insert(0, 2, -1.000);
  system.A.insert(0, 5, -1.000);
  system.A.insert(1, 0, -1.000);
  system.A.insert(1, 1, 4.000);
  system.A.insert(1, 2, -1.000);
  system.A.insert(1, 4, -1.000);
  system.A.insert(1, 6, -1.000);
  system.A.insert(2, 0, -1.000);
  system.A.insert(2, 1, -1.000);
  system.A.insert(2, 2, 5.000);
  system.A.insert(2, 3, -1.000);
  system.A.insert(2, 5, -1.000);
  system.A.insert(2, 7, -1.000);
  system.A.insert(3, 2, -1.000);
  system.A.insert(3, 3, 4.100);
  system.A.insert(3, 4, -1.000);
  system.A.insert(3, 5, -1.000);
  system.A.insert(3, 6, -1.000);
  system.A.insert(4, 1, -1.000);
  system.A.insert(4, 3, -1.000);
  system.A.insert(4, 4, 4.500);
  system.A.insert(4, 5, -1.000);
  system.A.insert(4, 7, -1.000);
  system.A.insert(5, 0, -1.000);
  system.A.insert(5, 2, -1.000);
  system.A.insert(5, 3, -1.000);
  system.A.insert(5, 4, -1.000);
  system.A.insert(5, 5, 5.000);
  system.A.insert(5, 6, -1.000);
  system.A.insert(6, 1, -1.000);
  system.A.insert(6, 3, -1.000);
  system.A.insert(6, 5, -1.000);
  system.A.insert(6, 6, 3.500);
  system.A.insert(6, 7, -1.000);
  system.A.insert(7, 2, -1.000);
  system.A.insert(7, 4, -1.000);
  system.A.insert(7, 6, -1.000);
  system.A.insert(7, 7, 3.300);
  system.b.set(8, 5);
  system.b.insert(0, 0.100);
  system.b.insert(3, 0.100);
  system.b.insert(4, 0.500);
  system.b.insert(6, -0.500);
  system.b.insert(7, 0.300);
  {
    system.x.set(8, 8);
    SparseGaussSeidelSolverd solver(1000, 1e-4, 4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
  {
    system.x.set(8, 8);
    SparseSORSolverd solver(1.79, 1000, 1e-4, 4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
  {
    system.x.set(8, 8);
    SparseCGSolverd<NullCGPreconditioner<SparseBlas2d>> solver(8, 1e-4);
    bool r = solver.solve(&system);
    system.x.iterate(
        [](const double &v, size_t i) { std::cout << v << std::endl; });
    std::cout << solver.lastNumberOfIterations << std::endl;
    std::cout << solver.lastResidual << std::endl;
    EXPECT_EQ(r, true);
  }
}
