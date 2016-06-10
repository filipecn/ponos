#pragma once

#include <iostream>
#include <vector>

namespace poseidon {

  class ConjugateGradient {
  public:
    ConjugateGradient() {}

    void set(uint32_t w, uint32_t h);
    void reset();
    void setup();
    void solve();
    void applyA();
    void applyPreconditioner();

    void printA() {
      for(int i = 0; i < width; i++) {
        for (int j = 0; j < height; ++j)
        {
          std::cout << "["
          << A[i][j].diag << " "
          << A[i][j].plusi << " "
          << A[i][j].plusj << " "
          << A[i][j].precon << "] ";
        }
        std::cout << std::endl;
      }
    }

    int ij(uint32_t i, uint32_t j) const {
      return i * width + j;
    }
    float& rhs(uint32_t i, uint32_t j) {
      return b[ij(i, j)];
    }

    float& Adiag(uint32_t i, uint32_t j) {
      return A[i][j].diag;
    }

    float& Aplusi(uint32_t i, uint32_t j) {
      return A[i][j].plusi;
    }

    float& Aplusj(uint32_t i, uint32_t j) {
      return A[i][j].plusj;
    }

    float P(uint32_t i, uint32_t j) const {
      return p[ij(i, j)];
    }

    float& precon(uint32_t i, uint32_t j) {
      return A[i][j].precon;
    }

    float& Q(uint32_t i, uint32_t j) {
      return q[ij(i, j)];
    }

    float& Z(uint32_t i, uint32_t j) {
      return z[ij(i, j)];
    }

    float& R(uint32_t i, uint32_t j) {
      return r[ij(i, j)];
    }

    uint32_t width, height;
    int max_iterations;
    float tol;
    std::vector<float> p, b;
    std::vector<float> r, z, q, s;

    struct ACell {
      float diag;
      float plusi;
      float plusj;
      float precon;
      ACell() {
        diag = plusi = plusj = precon = 0.f;
      }
    };

    std::vector<std::vector<ACell> > A;
  };

} // poseidon namespace
