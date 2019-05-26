#include <gtest/gtest.h>
#include <poseidon/poseidon.h>
#include <thrust/device_vector.h>

using namespace hermes::cuda;
using namespace poseidon::cuda;

TEST(SmokeSolver3, PressureSystem) {
  // y
  // |
  //  ---x
  // z = 0
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  // z = 1
  //  S S S S  - - - -
  //  S F F S  - 2 3 -
  //  S F F S  - 0 1 -
  //  S S S S  - - - -
  // z = 2
  //  S S S S  - - - -
  //  S F F S  - 6 7 -
  //  S F F S  - 4 5 -
  //  S S S S  - - - -
  // z = 3
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  //  S S S S  - - - -
  vec3u size(4);
  RegularGrid3Huc h_solid(size);
  RegularGrid3Hf h_div(size);
  auto hsAcc = h_solid.accessor();
  auto dAcc = h_div.accessor();
  int divIndex = 0;
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++) {
        dAcc(i, j, k) = 0;
        if (i > 0 && i < size.x - 1 && j > 0 && j < size.y - 1 && k > 0 &&
            k < size.z - 1) {
          dAcc(i, j, k) = divIndex++;
          hsAcc(i, j, k) = 0;
        } else
          hsAcc(i, j, k) = 1;
      }
  RegularGrid3Duc d_solid(size);
  memcpy(d_solid.data(), h_solid.data());
  RegularGrid3Df d_div(size);
  d_div.setSpacing(vec3f(0.01f));
  memcpy(d_div.data(), h_div.data());
  FDMatrix3D d_A(size);
  MemoryBlock1Dd d_rhs;
  size_t systemSize = setupPressureSystem(d_div, d_solid, d_A, 0.001, d_rhs);
  MemoryBlock1Hd h_rhs(d_rhs.size());
  h_rhs.allocate();
  memcpy(h_rhs, d_rhs);
  std::cerr << h_rhs << std::endl;
  FDMatrix3H h_A(size);
  h_A.copy(d_A);
  std::cerr << h_A << std::endl;
}

TEST(SmokeSolver3, pressureSystem) {
  // y
  // |
  //  ---x
  // z = 0
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  // z = 1
  //  S S S S S  -  -  -  -  -
  //  S F F F S  -  6  7  8  -
  //  S F F F S  -  3  4  5  -
  //  S F F F S  -  0  1  2  -
  //  S S S S S  -  -  -  -  -
  // z = 2
  //  S S S S S  -  -  -  -  -
  //  S F F F S  - 15 16 17  -
  //  S F F F S  - 12 13 14  -
  //  S F F F S  -  9 10 11  -
  //  S S S S S  -  -  -  -  -
  // z = 3
  //  S S S S S  -  -  -  -  -
  //  S F F F S  - 24 25 26  -
  //  S F F F S  - 21 22 23  -
  //  S F F F S  - 18 19 20  -
  //  S S S S S  -  -  -  -  -
  // z = 4
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  vec3u size(5);
  RegularGrid3Huc h_solid(size);
  auto hsAcc = h_solid.accessor();
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        if (i > 0 && i < size.x - 1 && j > 0 && j < size.y - 1 && k > 0 &&
            k < size.z - 1)
          hsAcc(i, j, k) = 0;
        else
          hsAcc(i, j, k) = 1;
  GridSmokeSolver3 solver;
  solver.setResolution(ponos::uivec3(size.x, size.y, size.z));
  solver.setSpacing(ponos::vec3f(0.01));
  solver.init();
  memcpy(solver.solid().data(), h_solid.data());
  { // setup velocity
    vec3u vSize(size.x, size.y + 1, size.z);
    MemoryBlock3Hf v(vSize);
    v.allocate();
    auto acc = v.accessor();
    for (int k = 0; k < vSize.z; k++)
      for (int j = 0; j < vSize.y; j++)
        for (int i = 0; i < vSize.x; i++)
          acc(i, j, k) = 0;
    acc(2, 3, 2) = -1;
    memcpy(solver.velocity().v().data(), v);
  }
  computeDivergence(solver.velocity(), solver.solid(), solver.divergence());
  std::cerr << solver.velocity().v().data() << std::endl;
  std::cerr << solver.divergence().data() << std::endl;
  FDMatrix3D A;
  A.resize(size);
  MemoryBlock1Dd rhs;
  setupPressureSystem(solver.divergence(), solver.solid(), A, 0.001, rhs);
  FDMatrix3H h_A(size);
  h_A.copy(A);
  // check symmetry
  auto acc = h_A.accessor();
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        for (int k2 = 0; k2 < size.z; k2++)
          for (int j2 = 0; j2 < size.y; j2++)
            for (int i2 = 0; i2 < size.x; i2++) {
              EXPECT_NEAR(acc(i, j, k, i2, j2, k2), acc(i2, j2, k2, i, j, k),
                          1e-8);
            }
  std::cerr << acc << std::endl;
  std::cerr << "solve\n";
  MemoryBlock1Dd x(rhs.size(), 0.f);
  pcg(x, A, rhs, rhs.size(), 1e-8);
}

TEST(SmokeSolver3, step) {
  // y
  // |
  //  ---x
  // z = 0
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  // z = 1
  //  S S S S S  -  -  -  -  -
  //  S F F F S  -  6  7  8  -
  //  S F F F S  -  3  4  5  -
  //  S F F F S  -  0  1  2  -
  //  S S S S S  -  -  -  -  -
  // z = 2
  //  S S S S S  -  -  -  -  -
  //  S F F F S  - 15 16 17  -
  //  S F F F S  - 12 13 14  -
  //  S F F F S  -  9 10 11  -
  //  S S S S S  -  -  -  -  -
  // z = 3
  //  S S S S S  -  -  -  -  -
  //  S F F F S  - 24 25 26  -
  //  S F F F S  - 21 22 23  -
  //  S F F F S  - 18 19 20  -
  //  S S S S S  -  -  -  -  -
  // z = 4
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  //  S S S S S  -  -  -  -  -
  vec3u size(5);
  RegularGrid3Huc h_solid(size);
  RegularGrid3Hf h_div(size);
  auto hsAcc = h_solid.accessor();
  for (int k = 0; k < size.z; k++)
    for (int j = 0; j < size.y; j++)
      for (int i = 0; i < size.x; i++)
        if (i > 0 && i < size.x - 1 && j > 0 && j < size.y - 1 && k > 0 &&
            k < size.z - 1)
          hsAcc(i, j, k) = 0;
        else
          hsAcc(i, j, k) = 1;
  GridSmokeSolver3 solver;
  solver.setResolution(ponos::uivec3(size.x, size.y, size.z));
  solver.setSpacing(ponos::vec3f(0.01));
  solver.init();
  memcpy(solver.solid().data(), h_solid.data());
  std::cerr << solver.solid().data() << std::endl;
  fill3(solver.scalarField(0), bbox3f(point3f(0.015f), point3f(0.025f)), 1.f,
        true);
  std::cerr << solver.scalarField(0).data() << std::endl;
  for (int i = 0; i < 10; i++)
    solver.step(0.01);
}

TEST(SmokeSolver3, step2) {
  return;
  int resSize = 128;
  ponos::uivec3 resolution(resSize);
  hermes::cuda::vec3u res(resSize);
  poseidon::cuda::GridSmokeSolver3 solver;
  solver.setSpacing(ponos::vec3f(1.f / res.x, 1.f / res.z, 1.f / res.z));
  solver.setResolution(resolution);
  solver.init();
  solver.rasterColliders();
  hermes::cuda::fill3(solver.scene().smoke_source.data().accessor(),
                      (unsigned char)0);
  poseidon::cuda::GridSmokeInjector3::injectSphere(ponos::point3f(0.35f), .15f,
                                                   solver.scalarField(0));
  solver.step(0.01);
}