#include <gtest/gtest.h>
#include <ponos/ponos.h>

using namespace ponos;

TEST(DirectSolvers, gaussj) {
  {
    DenseLinearSystemf system;
    system.A.set(2, 2);
    system.A(0, 0) = -1;
    system.A(0, 1) = 4;
    system.A(1, 0) = 0.2;
    system.A(1, 1) = -8;
    system.b.set(2);
    system.b[0] = 2;
    system.b[1] = 10;
    GaussJordanSolver<DenseLinearSystemf> solver;
    bool r = solver.solve(&system);
    EXPECT_EQ(r, true);
  }
  DenseLinearSystemf system;
  system.A.set(8, 8);
  system.A(0, 0) = 3.1;
  system.A(0, 1) = -1.0;
  system.A(0, 2) = -1.0;
  system.A(0, 5) = -1.0;
  system.A(1, 0) = -1.0;
  system.A(1, 1) = 4.0;
  system.A(1, 2) = -1.0;
  system.A(1, 4) = -1.0;
  system.A(1, 6) = -1.0;
  system.A(2, 0) = -1.0;
  system.A(2, 1) = -1.0;
  system.A(2, 2) = 5.0;
  system.A(2, 3) = -1.0;
  system.A(2, 5) = -1.0;
  system.A(2, 7) = -1.0;
  system.A(3, 2) = -1.0;
  system.A(3, 3) = 4.1;
  system.A(3, 4) = -1.0;
  system.A(3, 5) = -1.0;
  system.A(3, 6) = -1.0;
  system.A(4, 1) = -1.0;
  system.A(4, 3) = -1.0;
  system.A(4, 4) = 4.5;
  system.A(4, 5) = -1.0;
  system.A(4, 7) = -1.0;
  system.A(5, 0) = -1.0;
  system.A(5, 2) = -1.0;
  system.A(5, 3) = -1.0;
  system.A(5, 4) = -1.0;
  system.A(5, 5) = 5.0;
  system.A(5, 6) = -1.0;
  system.A(6, 1) = -1.0;
  system.A(6, 3) = -1.0;
  system.A(6, 5) = -1.0;
  system.A(6, 6) = 3.5;
  system.A(6, 7) = -1.0;
  system.A(7, 2) = -1.0;
  system.A(7, 4) = -1.0;
  system.A(7, 6) = -1.0;
  system.A(7, 7) = 3.3;
  system.b.set(8);
  system.b[0] = 0.1;
  system.b[3] = 0.1;
  system.b[4] = 0.5;
  system.b[6] = -0.5;
  system.b[7] = 0.3;
  {
    system.x.set(8);
    DenseMatrix<float> Acopy = system.A;
    GaussJordanSolver<DenseLinearSystemf> solver;
    bool r = solver.solve(&system);
    EXPECT_EQ(r, true);
  }
}
