#include "math/conjugate_gradient.h"

#include <algorithm>
#include <cstdlib>
#include <ponos.h>

extern "C" {
#include <cblas.h>
}

namespace poseidon {

  void ConjugateGradient::set(uint32_t w, uint32_t h) {
    width = w;
    height = h;
    uint32_t size = w * h;
    p.resize(size);
    b.resize(size);
    r.resize(size);
    z.resize(size);
    s.resize(size);
    q.resize(size);
    A.resize(w, std::vector<ACell>(h));
    max_iterations = 100;
    tol = 1e-6;
  }

  void ConjugateGradient::reset() {
    const ACell emptyCell;
    for(uint32_t i = 0; i < A.size(); ++i)
      std::fill(std::begin(A[i]), std::end(A[i]), emptyCell);
  }

  void ConjugateGradient::setup() {
    // start with guess p = 0
    std::fill(std::begin(p), std::end(p), 0.f);
    std::fill(std::begin(q), std::end(q), 0.f);
    std::fill(std::begin(z), std::end(z), 0.f);
    // set residual vector r = b
    cblas_scopy(b.size(), &b[0], 1, &r[0], 1);
  }

  void ConjugateGradient::applyPreconditioner() {
    cblas_scopy(z.size(), &r[0], 1, &z[0], 1);
    return;
    // set up the MIC(0) preconditioner
    precon(0, 0) = 0.f;
    float tal = 0.97f, sigma = 0.25f;
    for(uint32_t i = 1; i < width; ++i)
    for(uint32_t j = 1; j < height; j++) {
      //if(cell(i, j) == FLUID) {
      float e = Adiag(i, j)
      - SQR(Aplusi(i - 1, j) * precon(i - 1, j))
      - SQR(Aplusj(i, j - 1) * precon(i, j - 1))
      - tal * (Aplusi(i - 1, j) * Aplusj(i - 1, j) * SQR(precon(i - 1, j))
             + Aplusj(i, j - 1) * Aplusi(i, j - 1) * SQR(precon(i, j - 1)));
      if(e < sigma * Adiag(i, j))
      e = Adiag(i, j);
      precon(i, j) = 1.f / sqrtf(e);
      //}
    }

    // apply the preconditioner
    // first solve Lq = r
    Q(0, 0) = 0.f;
    for(uint32_t i = 1; i < width; ++i)
    for(uint32_t j = 1; j < height; j++) {
      //if(cell(i, j) == FLUID) {
      float t = R(i, j)
      - Aplusi(i - 1, j) * precon(i - 1, j) * Q(i - 1, j)
      - Aplusj(i, j - 1) * precon(i, j - 1) * Q(i, j - 1);
      Q(i, j) = t * precon(i, j);
      //}
    }
    // solve L^Tz = q
    for(int i = static_cast<int>(width) - 2; i >= 0; --i)
    for(int j = static_cast<int>(height) - 2; j >= 0; --j) {
      //if(cell(i, j) == FLUID) {
      float t = Q(i, j)
      - Aplusi(i, j) * precon(i, j) * Z(i + 1, j)
      - Aplusj(i, j) * precon(i, j) * Z(i, j + 1);
      Z(i, j) = t * precon(i, j);
      //}
    }

  }

  // setup() must be called before solve()
  void ConjugateGradient::solve() {
    setup();
    if (cblas_dsdot(b.size(), &b[0], 1, &b[0], 1) <= 1e-6)
      return;
    applyPreconditioner();
    // set seach vector s = z
    cblas_scopy(z.size(), &z[0], 1, &s[0], 1);
    // std::cout << "S = Z" << std::endl;
    // DUMP_VECTOR(s);
    // DUMP_VECTOR(z);
    double sigma = cblas_dsdot(z.size(), &z[0], 1, &r[0], 1);
    for (int i = 0; i < max_iterations; ++i) {
      // z = apply(A, s);
      // std::cout << "Z\n";
      // DUMP_VECTOR(z);
      // std::cout << "S\n";
      // DUMP_VECTOR(s);
      // std::cout << "MULT\n";
      applyA();
      // std::cout << "Z\n";
      // DUMP_VECTOR(z);
      double alpha = sigma / cblas_dsdot(z.size(), &z[0], 1, &s[0], 1);
      // std::cout << "alpha " << alpha << std::endl;
      // p = p + alpha * s (y = a*x + y)
      cblas_saxpy(s.size(), alpha, &s[0], 1, &p[0], 1);
      // std::cout << "P\n";
      // DUMP_VECTOR(p);
      // r = r - alpha * z
      // std::cout << "r = r - alpha * z\n";
      // DUMP_VECTOR(r);
      // DUMP_VECTOR(z);
      cblas_saxpy(z.size(), -alpha, &z[0], 1, &r[0], 1);
      // std::cout << "R\n";
      // DUMP_VECTOR(r);
      // std::cout << "Z\n";
      // DUMP_VECTOR(z);
      if (r[cblas_isamax(r.size(), &r[0], 1)] <= tol)
        return; // return p
      applyPreconditioner();
      double sigma_new = cblas_dsdot(z.size(), &z[0], 1, &r[0], 1);
      double beta = sigma_new / sigma;
      // std::cout << sigma_new << " ------ " << beta << std::endl;
      // s = z + beta * s
      cblas_sscal(s.size(), beta, &s[0], 1);
      // DUMP_VECTOR(s);
      cblas_saxpy(s.size(), 1.f, &z[0], 1, &s[0], 1);
      // DUMP_VECTOR(s);
      sigma = sigma_new;
      // std::cout << "SIGMA " << sigma << std::endl;
    }
  }

  void ConjugateGradient::applyA() {
    // z = As
    for(uint32_t i = 0; i < width; ++i)
    for(uint32_t j = 0; j < height; ++j) {
      uint32_t ind = ij(i, j);
      z[ind]  = A[i][j].diag * s[ind];
      if (i > 0)
        z[ind] += A[i][j].plusi * s[ij(i - 1, j)];
      if (i < width - 1)
        z[ind] += A[i][j].plusi * s[ij(i + 1, j)];
      if (j > 0)
        z[ind] += A[i][j].plusj * s[ij(i, j - 1)];
      if (j < height - 1)
        z[ind] += A[i][j].plusj * s[ij(i, j + 1)];
    }
  }

} // poseidon namespace
