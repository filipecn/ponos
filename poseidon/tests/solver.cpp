#include "solver.h"

void Solver::set(uint32_t n){
  B = Eigen::VectorXf(n);
  X = Eigen::VectorXf(n);
  A = Eigen::SparseMatrix<float>(n,n);
  A.reserve(Eigen::VectorXi::Constant(n,5));
}

void Solver::solve(){
  Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > cg;
  cg.compute(A);
  X = cg.solve(B);
  //std::cout << "#iterations:     " << cg.iterations() << std::endl;
  //std::cout << "estimated error: " << cg.error()      << std::endl;
  //std::cout << B << std::endl;
  //std::cout << "_____\n";
  //std::cout << X << std::endl;
  //std::cout << A << std::endl;
}
