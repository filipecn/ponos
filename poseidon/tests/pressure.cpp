#include <poseidon.h>
#include <ponos.h>

#include "solver.h"

poseidon::ConjugateGradient ps;
Solver s;
int width = 5, height = 5;
int ij(int i, int j) {return i*width + j;}

void solveCG() {
  ps.solve();

  DUMP_VECTOR(ps.p);
}

int main() {
  ps.set(width, height);
  s.set(width * height);
  ps.reset();
  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++) {
      // A
      ps.Adiag(i, j) = 1.f;
      ps.Aplusi(i, j) - 0.f;
      ps.Aplusj(i, j) - 0.f;
      s.A.coeffRef(ij(i,j),ij(i,j)) = 1.0f;
      // B
      ps.rhs(i, j) = i;
      s.B(ij(i, j)) = i;
    }
  s.solve();
  for(int i = 0; i < width * height; i++)
    std::cout << s.X(i) << " ";
    std::cout << std::endl;

  solveCG();
  return 0;
}
