#ifndef POSEIDON_ALGORITHMS_SOLVER_H
#define POSEIDON_ALGORITHMS_SOLVER_H

#include <Eigen/Sparse>

#include <ponos.h>

namespace poseidon {

class Solver {
public:
  Solver() {}
  ~Solver() {}

  void incrementB(int i, float v) { B.coeffRef(i) += v; }

  void setB(int i, float v) { B.coeffRef(i) = v; }

  void setA(int i, int j, float v) { A.coeffRef(i, j) = v; }

  float getX(int i) { return X(i); }

  void set(uint n) {
    size = n;
    B = Eigen::VectorXf(n);
    X = Eigen::VectorXf(n);
    A = Eigen::SparseMatrix<float>(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 5));
  }

  void solve() {
    // compute the Cholesky decomposition of A
    // Eigen::LLT<Eigen::MatrixXd> lltOfA(A);
    // ASSERT_FATAL(lltOfA.info() != Eigen::NumericalIssue);
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> cg;
    cg.compute(A);
    X = cg.solve(B);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error() << std::endl;
  }

  void reset() {
    set(size);
    B.setZero();
  }

  // private:
  uint size;
  Eigen::VectorXf B;
  Eigen::VectorXf X;
  Eigen::SparseMatrix<float> A;
};

} // poseidon namespace

#endif // POSEIDON_ALGORITHMS_SOLVER_H
