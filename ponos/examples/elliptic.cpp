/* Solving the poisson equation d2u/h2 + d2u/dy2 = -2pi^2cos(pix)cos(piy)
 * R = {(x,y) E R2, a < x < b, c < y < d}
 * command lines: h <solveType SOR|cg> <gridtype v|c>
 */

#include <ponos.h>

using namespace ponos;

SparseLinearSystemd ls;
int N, M, size;
double a, b, c, d, h;
std::function<double(int)> posX, posY;
double PI2 = SQR(PI);

double f(double x, double y) { return -2.0 * PI2 * cos(PI * x) * cos(PI * y); }

double analyticSolution(double x, double y) {
  return cos(PI * x) * cos(PI * y);
}

void buildVertexCenteredSystem() {
  size = (N - 1) * (M - 1);
  auto inside = [](int i, int j) -> int {
    return (i >= 1 && i < M && j >= 1 && j < N) ? 1 : 0;
  };
  auto ind = [](int i, int j) -> int { return (j - 1) * (M - 1) + (i - 1); };
  posX = [](int i) -> double { return a + i * h; };
  posY = [](int j) -> double { return c + j * h; };
  ls.A.set(size, size, size * 5);
  ls.b.set(size, size);
  for (int j = 1; j < N; j++) {
    for (int i = 1; i < M; i++) {
      /*std::cout << (inside(i, j - 1) ? ind(i, j - 1) : -1) << " "
                << (inside(i - 1, j) ? ind(i - 1, j) : -1) << " "
                << (inside(i, j) ? ind(i, j) : -1) << " "
                << (inside(i + 1, j) ? ind(i + 1, j) : -1) << " "
                << (inside(i, j + 1) ? ind(i, j + 1) : -1) << std::endl;*/
      double rhs = f(posX(i), posY(j));
      // ui,j-1
      if (j - 1 > 0)
        ls.A.insert(ind(i, j), ind(i, j - 1), 1);
      // ui-1,j
      if (i - 1 == 0)
        rhs -= cos(PI * posY(j)) / SQR(h);
      else
        ls.A.insert(ind(i, j), ind(i - 1, j), 1);
      // uij
      ls.A.insert(ind(i, j), ind(i, j),
                  -4.0 + !inside(i, j - 1) + !inside(i, j + 1));
      // ui+1,j
      if (i + 1 == M)
        rhs += cos(PI * posY(j)) / SQR(h);
      else
        ls.A.insert(ind(i, j), ind(i + 1, j), 1);
      // ui,j+1
      if (j + 1 < N)
        ls.A.insert(ind(i, j), ind(i, j + 1), 1);
      ls.b.insert(ind(i, j), rhs * SQR(h));
    }
  }
  // ls.A.printDense();
  // std::cout << ls.b;
}

void buildCellCenteredSystem() {
  size = N * M;
  auto inside = [](int i, int j) -> int {
    return (i >= 1 && i <= M && j >= 1 && j <= N) ? 1 : 0;
  };
  auto ind = [](int i, int j) -> int { return (j - 1) * M + (i - 1); };
  posX = [](int i) -> double { return a + i * h - h * 0.5; };
  posY = [](int j) -> double { return c + j * h - h * 0.5; };
  ls.A.set(size, size, size * 5);
  ls.b.set(size, size);
  for (int j = 1; j <= N; j++) {
    for (int i = 1; i <= M; i++) {
      // std::cout << (inside(i, j - 1) ? ind(i, j - 1) : -1) << " "
      //          << (inside(i - 1, j) ? ind(i - 1, j) : -1) << " "
      //          << (inside(i, j) ? ind(i, j) : -1) << " "
      //          << (inside(i + 1, j) ? ind(i + 1, j) : -1) << " "
      //          << (inside(i, j + 1) ? ind(i, j + 1) : -1) << std::endl;
      double rhs = f(posX(i), posY(j));
      // ui,j-1
      if (j - 1 > 0)
        ls.A.insert(ind(i, j), ind(i, j - 1), 1);
      // ui-1,j
      if (i - 1 == 0)
        rhs -= 2.0 * cos(PI * posY(j)) / SQR(h);
      else
        ls.A.insert(ind(i, j), ind(i - 1, j), 1);
      // uij
      ls.A.insert(ind(i, j), ind(i, j),
                  -4.0 - !inside(i - 1, j) - !inside(i + 1, j) +
                      !inside(i, j - 1) + !inside(i, j + 1));
      // ui+1,j
      if (i + 1 > M)
        rhs += 2.0 * cos(PI * posY(j)) / SQR(h);
      else
        ls.A.insert(ind(i, j), ind(i + 1, j), 1);
      // ui,j+1
      if (j + 1 <= N)
        ls.A.insert(ind(i, j), ind(i, j + 1), 1);
      ls.b.insert(ind(i, j), rhs * SQR(h));
    }
  }
  // ls.A.printDense();
  // std::cout << ls.b;
}

int main(int argc, char **argv) {
  a = 0, b = 1, c = 0, d = 1;
  h = 0.2;
  char solverType = 'c';
  char meshType = 'v';
  if (argc > 3) {
    sscanf(argv[1], "%lf", &h);
    solverType = argv[2][0];
    meshType = argv[3][0];
  }
  N = (d - c) / h; // y direction
  M = (b - a) / h; // x direction
  if (meshType == 'v')
    buildVertexCenteredSystem();
  else if (meshType == 'c')
    buildCellCenteredSystem();
  ls.x.set(size, size);
  double error = 0.0, time = 0.0, lastResidual = 10000;
  int iterations = 0;
  if (solverType == 'S') {
    SparseSORSolverd solver(1.79, size, 1e-8, 4);
    Timer timer;
    /*bool r = */ solver.solve(&ls);
    time = timer.tack();
    // std::cout << timer.tack() << std::endl;
    // std::cout << " it " << solver.lastNumberOfIterations << std::endl;
    // std::cout << " re " << solver.lastResidual << std::endl;
    iterations = solver.lastNumberOfIterations;
    lastResidual = solver.lastResidual;
    // std::cout << r << std::endl;
  } else if (solverType == 'c') {
    SparseCGSolverd<NullCGPreconditioner<SparseBlas2d>> solver(size, 1e-8);
    Timer timer;
    /*bool r = */ solver.solve(&ls);
    time = timer.tack();
    iterations = solver.lastNumberOfIterations;
    lastResidual = solver.lastResidual;
    // std::cout << " re " << solver.lastResidual << std::endl;
  }
  ls.x.iterate([](const double &v, size_t i) { std::cout << v << std::endl; });
  if (meshType == 'v') {
    SparseVector<double>::const_iterator it(ls.x);
    // std::cout << it.count() << " " << (N - 1) * (M - 1) << std::endl;
    ASSERT_FATAL(it.count() == (N - 1) * (M - 1));
    for (int j = 1; j < N; j++)
      for (int i = 1; i < M; i++) {
        double v = std::fabs(analyticSolution(posX(i), posY(j)) - it.value());
        std::cout << analyticSolution(posX(i), posY(j)) << std::endl;
        error = std::max(error, v);
        ++it;
      }
    // std::cout << std::endl;
    // std::cout << error << std::endl;

  } else if (meshType == 'c') {
    SparseVector<double>::const_iterator it(ls.x);
    // std::cout << it.count() << " " << (N - 1) * (M - 1) << std::endl;
    ASSERT_FATAL(it.count() == N * M);
    double n2 = 0.0;
    // std::cout << "EROOOOROR\n";
    for (int j = 1; j <= N; j++)
      for (int i = 1; i <= M; i++) {
        double v = std::fabs(analyticSolution(posX(i), posY(j)) - it.value());
        std::cout << analyticSolution(posX(i), posY(j)) << std::endl;
        // std::cout << v << std::endl;
        n2 += SQR(v);
        error = std::max(error, v);
        ++it;
      }
    // std::cout << std::endl;
    // std::cout << error << std::endl;
    // std::cout << std::sqrt(n2) << std::endl;
    // std::cout << r << std::endl;
  }
  std::cout << error << std::endl;
  std::cout << time << std::endl;
  std::cout << lastResidual << std::endl;
  std::cout << iterations << std::endl;
  return 0;
}
