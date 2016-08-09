#pragma once

#include <iostream>
#include <Eigen/Sparse>

  class Solver {
  public:
    Solver(){}
    ~Solver(){}

    void set(uint32_t n);
    void solve();

    Eigen::VectorXf B;
    Eigen::VectorXf X;
    Eigen::SparseMatrix<float> A;
  };
