#include <ponos.h>

using namespace ponos;

int N = 10;
double dx, dt, L = 2.0;
double ui0(int i) {
  if (i == 0 || i == N)
    return 0.0;
  return i * dx * (L - i * dx);
}
void computeExplicit(std::vector<double> &unext) {
  double sigma = dt / (dx * dx);
  unext.emplace_back(0);
  for (int i = 1; i < N; i++)
    unext.emplace_back(ui0(i) +
                       (ui0(i + 1) - 2.0 * ui0(i) + ui0(i - 1)) * sigma);
  unext.emplace_back(0);
}

void computeCrankNicolson(std::vector<double> &unext) {
  SparseLinearSystemd ls;
  int size = N - 1;
  ls.A.set(size, size, size * 3);
  ls.b.set(size, size);
  double sigma = dt / (2.0 * dx * dx);
  for (int i = 1; i < N; i++) {
    ls.b.insert(i - 1,
                ui0(i) + sigma * (ui0(i + 1) - 2.0 * ui0(i) + ui0(i - 1)));
    if (i > 1)
      ls.A.insert(i - 1, i - 2, -sigma);
    ls.A.insert(i - 1, i - 1, 1.0 + 2.0 * sigma);
    if (i < N - 1)
      ls.A.insert(i - 1, i, -sigma);
  }
  // ls.A.printDense();
  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << ls.b << std::endl;
  ls.x.set(size, size);
  SparseCGSolverd<NullCGPreconditioner<SparseBlas2d>> solver(size, 1e-8);
  if (!solver.solve(&ls))
    std::cout << "ops\n";
  double lastResidual = 10000;
  int iterations = 0;
  iterations = solver.lastNumberOfIterations;
  lastResidual = solver.lastResidual;
  // std::cout << " re " << solver.lastResidual << std::endl;
  unext.emplace_back(0);
  ls.x.iterate([&unext](const double &v, size_t i) {
    unext.emplace_back(v);
    // std::cout << v << std::endl;
  });
  unext.emplace_back(0);
}

void computePhi(const std::vector<double> &u, std::vector<double> &x) {
  SparseLinearSystemd ls;
  int size = N;
  ls.A.set(size, size, size * 3);
  ls.b.set(size, size);
  double sigma = 1.0 / (dx * dx);
  for (int i = 1; i <= N; i++) {
    ls.b.insert(i - 1, (u[i] - u[i - 1]) / dx);
    if (i > 1)
      ls.A.insert(i - 1, i - 2, sigma);
    if (i >= N)
      ls.A.insert(i - 1, i - 1, -3.0 * sigma);
    else if (i > 1)
      ls.A.insert(i - 1, i - 1, -2.0 * sigma);
    else
      ls.A.insert(i - 1, i - 1, -1.0 * sigma);
    if (i < N)
      ls.A.insert(i - 1, i, sigma);
  }
  // ls.A.printDense();
  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << ls.b << std::endl;
  ls.x.set(size, size);
  SparseCGSolverd<NullCGPreconditioner<SparseBlas2d>> solver(size, 1e-8);
  if (!solver.solve(&ls))
    std::cout << "ops\n";
  double lastResidual = 10000;
  int iterations = 0;
  iterations = solver.lastNumberOfIterations;
  lastResidual = solver.lastResidual;
  // std::cout << " re " << solver.lastResidual << std::endl;
  ls.x.iterate([&x](const double &v, size_t i) {
    x.emplace_back(v);
    // std::cout << v << std::endl;
  });
}

void computeU(std::vector<double> &u, const std::vector<double> &phi) {
  for (size_t i = 1; i < u.size() - 1; i++) {
    u[i] = u[i] - (phi[i] - phi[i - 1]) / dx;
  }
}

int main(int argc, char **argv) {
  dx = L / static_cast<double>(N);
  dt = 0.25 * dx * dx;
  std::vector<double> unext, unext_cn;
  std::vector<double> phi, phi_cn;
  computeExplicit(unext);
  computeCrankNicolson(unext_cn);
  computePhi(unext, phi);
  computePhi(unext_cn, phi_cn);
  std::cout << "x\tu_exp\tu_cn\tphi\n";
  for (size_t i = 0; i < unext.size(); i++) {
    std::cout << i * dx << "\t" << unext[i] << "\t" << unext_cn[i] << "\t";
    if (i >= 1)
      std::cout << phi[i] << std::endl;
    else
      std::cout << "xxxx" << std::endl;
  }
  computeU(unext, phi);
  computeU(unext_cn, phi_cn);
  return 0;
}
