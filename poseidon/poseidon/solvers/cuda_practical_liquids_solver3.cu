/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * iM the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <algorithm>
#include <hermes/numeric/cuda_blas.h>
#include <poseidon/math/cuda_pcg.h>
#include <poseidon/solvers/cuda_practical_liquids_solver3.h>
#include <queue>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

__global__ void __computeDistanceNextToSurface(LevelSet3Accessor in,
                                               LevelSet3Accessor out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (out.isIndexStored(i, j, k)) {
    float minDist = Constants::greatest<float>();
    int dir[6][3] = {{-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
                     {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
    for (int d = 0; d < 6; d++) {
      int I = i + dir[d][0];
      int J = j + dir[d][1];
      int K = k + dir[d][2];
      float in_ijk = in(i, j, k);
      float in_IJK = in(I, J, K);
      if (sign(in_ijk) * sign(in_IJK) < 0) {
        float theta = in_ijk / (in_ijk - in_IJK);
        // if ((i > I || j > J || k > K) && in_ijk > in_IJK)
        //   theta *= -1;
        float phi = sign(in(i, j, k)) * theta * in.spacing().x;
        if (fabsf(minDist) > fabs(phi))
          minDist = phi;
      }
    }
    out(i, j, k) = minDist;
  }
}

__global__ void __handleSolids(LevelSet3Accessor s, LevelSet3Accessor f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (f.isIndexStored(i, j, k)) {
    if (f(i, j, k) < 0 && s(i, j, k) < 0) {
      f(i, j, k) = -s(i, j, k);
    }
    // else if (f(i, j, k) < 0 && fabs(f(i, j, k)) > fabs(s(i, j, k))) {
    //   f(i, j, k) = sign(f(i, j, k)) * fabs(s(i, j, k));
    // }
  }
}

void propagate(LevelSet3H &ls, int s) {
  auto phi = ls.grid().accessor();
  MemoryBlock3Huc frozenGrid(phi.resolution());
  frozenGrid.allocate();
  auto frozen = frozenGrid.accessor();
  fill3(frozenGrid, (unsigned char)0);
  struct Point_ {
    Point_(int I, int J, int K, float D) : i(I), j(J), k(K), t(D) {}
    int i, j, k;
    float t; // tentative distance
  };
  auto cmp = [](Point_ a, Point_ b) { return a.t > b.t; };
  std::priority_queue<Point_, std::vector<Point_>, decltype(cmp)> q(cmp);
  // get initial set of points
  for (auto e : phi)
    if (fabsf(e.value) < Constants::greatest<float>() && sign(e.value) * s > 0)
      q.push(Point_(e.i(), e.j(), e.k(), fabsf(e.value)));
  while (!q.empty()) {
    Point_ p = q.top();
    q.pop();
    frozen(p.i, p.j, p.k) = 2;
    phi(p.i, p.j, p.k) = s * p.t;
    vec3i dir[6] = {vec3i(-1, 0, 0), vec3i(1, 0, 0),  vec3i(0, -1, 0),
                    vec3i(0, 1, 0),  vec3i(0, 0, -1), vec3i(0, 0, 1)};
    for (int direction = 0; direction < 6; direction++) {
      int i = p.i + dir[direction].x;
      int j = p.j + dir[direction].y;
      int k = p.k + dir[direction].z;
      if (!frozen.isIndexValid(i, j, k) || frozen(i, j, k) ||
          (fabs(phi(i, j, k)) < Constants::greatest<float>() &&
           sign(phi(i, j, k)) * s < 0))
        continue;
      float phis[3] = {
          fminf(fabsf(phi(p.i - 1, p.j, p.k)), fabsf(phi(p.i + 1, p.j, p.k))),
          fminf(fabsf(phi(p.i, p.j - 1, p.k)), fabsf(phi(p.i, p.j + 1, p.k))),
          fminf(fabsf(phi(p.i, p.j, p.k - 1)), fabsf(phi(p.i, p.j, p.k + 1)))};
      std::sort(phis, phis + 3);
      // try closest neighbor
      float d = phis[0] + phi.spacing().x; // TODO: assuming square cells
      if (d > phis[1]) {
        // try the two closest neighbor
        d = (phis[0] + phis[1] +
             sqrtf(2.f * phi.spacing().x * phi.spacing().x -
                   (phis[1] - phis[0]) * (phis[1] - phis[0]))) *
            0.5f;
        if (d > phis[2])
          // use ll three neighbors
          d = (phis[0] + phis[1] + phis[2] +
               sqrtf(
                   fmaxf(0.f, (phis[0] + phis[1] + phis[2]) *
                                      (phis[0] + phis[1] + phis[2]) -
                                  3.f * (phis[0] * phis[0] + phis[1] * phis[1] +
                                         phis[2] * phis[2] -
                                         phi.spacing().x * phi.spacing().x)))) /
              3.f;
      }
      phi(i, j, k) = s * fminf(fabsf(phi(i, j, k)), d);
      q.push(Point_(i, j, k, fabsf(phi(i, j, k))));
      frozen(i, j, k) = 1;
    }
  }
}

template <>
void PracticalLiquidsSolver3<
    hermes::cuda::MemoryLocation::DEVICE>::updateSurfaceDistanceField() {
  hermes::ThreadArrayDistributionInfo td(surface_ls_[DST].grid().resolution());
  // initiate all domain with infinity distance
  fill3(surface_ls_[DST].grid().data(), Constants::greatest<float>());
  __handleSolids<<<td.gridSize, td.blockSize>>>(solid_ls_.accessor(),
                                                surface_ls_[SRC].accessor());
  // std::cerr << "FLUID BEFORE\n" << surface_ls_[SRC].grid().data() <<
  // std::endl; compute distances of points closest to surface
  __computeDistanceNextToSurface<<<td.gridSize, td.blockSize>>>(
      surface_ls_[SRC].accessor(), surface_ls_[DST].accessor());
  // std::cerr << "FLUID\n" << surface_ls_[DST].grid().data() << std::endl;
  {
    LevelSet3H hls(surface_ls_[DST]);
    // propagate distances in positive direction
    propagate(hls, 1);
    // propagate distances in negative direction
    propagate(hls, -1);
    memcpy(surface_ls_[DST].grid().data(), hls.grid().data());
  }
  // std::cerr << "SOLID\n" << solid_ls_.grid().data() << std::endl;
  // std::cerr << "FLUID AFTER PROPAGATION\n"
  //           << surface_ls_[DST].grid().data() << std::endl;
  __handleSolids<<<td.gridSize, td.blockSize>>>(solid_ls_.accessor(),
                                                surface_ls_[DST].accessor());
  // std::cerr << "FLUID AFTER SOLID\n"
  //           << surface_ls_[DST].grid().data() << std::endl;
  // exit(0);
}

__global__ void __classifyCells(LevelSet3Accessor ls, LevelSet3Accessor sls,
                                RegularGrid3Accessor<MaterialType> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (m.isIndexStored(i, j, k)) {
    if (sls(i, j, k) <= 0)
      m(i, j, k) = MaterialType::SOLID;
    else if (ls(i, j, k) <= 0)
      m(i, j, k) = MaterialType::FLUID;
    else
      m(i, j, k) = MaterialType::AIR;
  }
}

template <>
void PracticalLiquidsSolver3<
    hermes::cuda::MemoryLocation::DEVICE>::classifyCells() {
  hermes::ThreadArrayDistributionInfo td(material_.resolution());
  __classifyCells<<<td.gridSize, td.blockSize>>>(
      surface_ls_[DST].accessor(), solid_ls_.accessor(), material_.accessor());
}

__global__ void __rasterSolid(LevelSet3Accessor ls, StaggeredGrid3Accessor vel,
                              bbox3f box, vec3f v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (ls.isIndexStored(i, j, k)) {
    ls(i, j, k) = fminf(ls(i, j, k), SDF::box(box, ls.worldPosition(i, j, k)));
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::rasterSolid(
    const bbox3f &box, const vec3f &velocity) {
  hermes::ThreadArrayDistributionInfo td(solid_ls_.grid().resolution());
  __rasterSolid<<<td.gridSize, td.blockSize>>>(
      solid_ls_.accessor(), solid_velocity_.accessor(), box, velocity);
}

__global__ void __applyExternalForces(RegularGrid3Accessor<float> in,
                                      RegularGrid3Accessor<float> out,
                                      VectorGrid3Accessor force,
                                      RegularGrid3Accessor<MaterialType> m,
                                      float dt, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (in.isIndexStored(i, j, k)) {
    if (m(i - (0 == dim), j - (1 == dim), k - (2 == dim)) ==
            MaterialType::FLUID &&
        m(i, j, k) == MaterialType::FLUID) {
      float f = (force(i - (0 == dim), j - (1 == dim), k - (2 == dim))[dim] +
                 force(i, j, k)[dim]) *
                0.5f;
      out(i, j, k) = in(i, j, k) + dt * f;
    }
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::applyExternalForces(
    float dt) {
  {
    hermes::ThreadArrayDistributionInfo td(velocity_[SRC].u().resolution());
    __applyExternalForces<<<td.gridSize, td.blockSize>>>(
        velocity_[SRC].u().accessor(), velocity_[DST].u().accessor(),
        force_field_.accessor(), material_.accessor(), dt, 0);
  }
  {
    hermes::ThreadArrayDistributionInfo td(velocity_[SRC].v().resolution());
    __applyExternalForces<<<td.gridSize, td.blockSize>>>(
        velocity_[SRC].v().accessor(), velocity_[DST].v().accessor(),
        force_field_.accessor(), material_.accessor(), dt, 1);
  }
  {
    hermes::ThreadArrayDistributionInfo td(velocity_[SRC].w().resolution());
    __applyExternalForces<<<td.gridSize, td.blockSize>>>(
        velocity_[SRC].w().accessor(), velocity_[DST].w().accessor(),
        force_field_.accessor(), material_.accessor(), dt, 2);
  }
}

__global__ void __computeDivergence(StaggeredGrid3Accessor vel,
                                    RegularGrid3Accessor<MaterialType> solid,
                                    RegularGrid3Accessor<float> divergence,
                                    vec3f invdx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (divergence.isIndexStored(i, j, k)) {
    float left = vel.u(i, j, k);
    float right = vel.u(i + 1, j, k);
    float bottom = vel.v(i, j, k);
    float top = vel.v(i, j + 1, k);
    float back = vel.w(i, j, k);
    float front = vel.w(i, j, k + 1);
    bool sleft = solid(i - 1, j, k) == MaterialType::SOLID;
    bool sright = solid(i + 1, j, k) == MaterialType::SOLID;
    bool sbottom = solid(i, j - 1, k) == MaterialType::SOLID;
    bool stop = solid(i, j + 1, k) == MaterialType::SOLID;
    bool sback = solid(i, j, k - 1) == MaterialType::SOLID;
    bool sfront = solid(i, j, k + 1) == MaterialType::SOLID;
    if (sleft)
      left = 0;
    if (sright)
      right = 0;
    if (sbottom)
      bottom = 0;
    if (stop)
      top = 0;
    if (sback)
      back = 0;
    if (sfront)
      front = 0;
    divergence(i, j, k) =
        dot(invdx, vec3f(right - left, top - bottom, front - back));
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::computeDivergence() {
  vec3f inv(-1.f / divergence_.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence_.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      velocity_[DST].accessor(), material_.accessor(), divergence_.accessor(),
      inv);
}

__global__ void __fillPressureMatrix(MemoryBlock3Accessor<FDMatrix3Entry> A,
                                     RegularGrid3Accessor<MaterialType> m,
                                     float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (A.isIndexValid(i, j, k)) {
    if (m(i, j, k) == MaterialType::SOLID)
      return;
    A(i, j, k).diag = 0;
    A(i, j, k).x = 0;
    A(i, j, k).y = 0;
    A(i, j, k).z = 0;
    if (m.isIndexStored(i - 1, j, k) && m(i - 1, j, k) != MaterialType::SOLID)
      A(i, j, k).diag += scale;
    if (m.isIndexStored(i + 1, j, k) && m(i + 1, j, k) == MaterialType::FLUID) {
      A(i, j, k).diag += scale;
      A(i, j, k).x = -scale;
    } else if (m.isIndexStored(i + 1, j, k) &&
               m(i + 1, j, k) == MaterialType::AIR)
      A(i, j, k).diag += scale;
    if (m.isIndexStored(i, j - 1, k) && m(i, j - 1, k) != MaterialType::SOLID)
      A(i, j, k).diag += scale;
    if (m.isIndexStored(i, j + 1, k) && m(i, j + 1, k) == MaterialType::FLUID) {
      A(i, j, k).diag += scale;
      A(i, j, k).y = -scale;
    } else if (m.isIndexStored(i, j + 1, k) &&
               m(i, j + 1, k) == MaterialType::AIR)
      A(i, j, k).diag += scale;
    if (m.isIndexStored(i, j, k - 1) && m(i, j, k - 1) != MaterialType::SOLID)
      A(i, j, k).diag += scale;
    if (m.isIndexStored(i, j, k + 1) && m(i, j, k + 1) == MaterialType::FLUID) {
      A(i, j, k).diag += scale;
      A(i, j, k).z = -scale;
    } else if (m.isIndexStored(i, j, k + 1) &&
               m(i, j, k + 1) == MaterialType::AIR)
      A(i, j, k).diag += scale;
  }
}

__global__ void __buildRHS(MemoryBlock3Accessor<int> indices,
                           RegularGrid3Accessor<float> divergence,
                           MemoryBlock1Accessor<double> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (indices.isIndexValid(i, j, k)) {
    if (indices(i, j, k) >= 0)
      rhs[indices(i, j, k)] = divergence(i, j, k);
  }
}

__global__ void __1To3(MemoryBlock3Accessor<int> indices,
                       MemoryBlock1Accessor<double> v,
                       MemoryBlock3Accessor<float> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (indices.isIndexValid(i, j, k)) {
    if (indices(i, j, k) >= 0)
      m(i, j, k) = v[indices(i, j, k)];
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::solvePressure(float dt) {
  // fill matrix
  float scale = dt / (divergence_.spacing().x * divergence_.spacing().x);
  {
    hermes::ThreadArrayDistributionInfo td(divergence_.resolution());
    __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
        pressure_matrix_.dataAccessor(), material_.accessor(), scale);
  }
  // compute indices
  auto res = divergence_.resolution();
  MemoryBlock3Hi h_indices(res);
  h_indices.allocate();
  MemoryBlock3<MemoryLocation::HOST, MaterialType> h_material(res);
  h_material.allocate();
  memcpy(h_material, material_.data());
  auto m = h_material.accessor();
  int curIndex = 0;
  for (auto e : h_indices.accessor())
    if (m(e.i(), e.j(), e.k()) == MaterialType::FLUID)
      e.value = curIndex++;
    else
      e.value = -1;
  memcpy(pressure_matrix_.indexData(), h_indices);
  // rhs
  MemoryBlock1Dd rhs;
  rhs.resize(curIndex);
  rhs.allocate();
  {
    hermes::ThreadArrayDistributionInfo td(divergence_.resolution());
    __buildRHS<<<td.gridSize, td.blockSize>>>(
        pressure_matrix_.indexDataAccessor(), divergence_.accessor(),
        rhs.accessor());
  }
  // apply incomplete Cholesky preconditioner
  // solve system
  MemoryBlock1Dd x(rhs.size(), 0.f);
  std::cerr << "solve\n";
  if (pcg(x, pressure_matrix_, rhs, rhs.size(), 1e-8)) {
    // std::cerr << material_.data() << std::endl;
    // std::cerr << velocity_[DST].u().data() << std::endl;
    // std::cerr << velocity_[DST].v().data() << std::endl;
    // std::cerr << divergence_.data() << std::endl;
    // exit(-1);
    // std::cerr << rhs << std::endl;
    // FDMatrix2H H(pressure_matrix_.gridSize());
    // H.copy(pressure_matrix_);
    // auto acc = H.accessor();
    // std::cerr << acc << std::endl;
    // std::cerr << pressure_matrix_.indexData() << std::endl;
  }
  // std::cerr << residual << "\n" << x << std::endl;
  // store pressure values
  MemoryBlock1Dd sol(rhs.size(), 0);
  mul(pressure_matrix_, x, sol);
  sub(sol, rhs, sol);
  if (infnorm(sol, sol) > 1e-6) {
    FDMatrix3H H(pressure_matrix_.gridSize());
    H.copy(pressure_matrix_);
    auto acc = H.accessor();
    std::cerr << material_.data() << std::endl;
    std::cerr << pressure_matrix_.indexData() << std::endl;
    std::cerr << acc << std::endl;
    std::cerr << "WRONG PCG!\n";
    exit(-1);
    // std::cerr << sol << std::endl;
  }
  {
    hermes::ThreadArrayDistributionInfo td(pressure_.resolution());
    __1To3<<<td.gridSize, td.blockSize>>>(pressure_matrix_.indexDataAccessor(),
                                          x.accessor(),
                                          pressure_.data().accessor());
  }
}

__global__ void __projectionStep(RegularGrid3Accessor<float> vel,
                                 RegularGrid3Accessor<float> pressure,
                                 RegularGrid3Accessor<MaterialType> m, vec3u d,
                                 float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (vel.isIndexStored(i, j, k)) {
    if (m(i - d.x, j - d.y, k - d.z) == MaterialType::SOLID)
      vel(i, j, k) = 0; // tex3D(wSolidTex3, xc, yc, zc - 1);
    else if (m(i, j, k) == MaterialType::SOLID)
      vel(i, j, k) = 0; // tex3D(wSolidTex3, xc, yc, zc);
    else {
      float b = pressure(i - d.x, j - d.y, k - d.z);
      float f = pressure(i, j, k);
      vel(i, j, k) -= scale * (f - b);
    }
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::project(float dt) {
  {
    auto info = velocity_[DST].u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity_[DST].u().accessor(), pressure_.accessor(),
        material_.accessor(), vec3u(1, 0, 0), scale);
  }
  {
    auto info = velocity_[DST].v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity_[DST].v().accessor(), pressure_.accessor(),
        material_.accessor(), vec3u(0, 1, 0), scale);
  }
  {
    auto info = velocity_[DST].w().info();
    float invdx = 1.0 / info.spacing.z;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity_[DST].w().accessor(), pressure_.accessor(),
        material_.accessor(), vec3u(0, 0, 1), scale);
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::advectVelocity(float dt) {
  SemiLagrangianIntegrator3 integrator;
  integrator.advect(velocity_[SRC], material_, velocity_[SRC].u(),
                    velocity_[DST].u(), dt);
  integrator.advect(velocity_[SRC], material_, velocity_[SRC].v(),
                    velocity_[DST].v(), dt);
  integrator.advect(velocity_[SRC], material_, velocity_[SRC].w(),
                    velocity_[DST].w(), dt);
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::advectFluid(float dt) {
  fluidIntegrator.advect(velocity_[SRC], material_, surface_ls_[SRC].grid(),
                         surface_ls_[DST].grid(), dt);
  // SemiLagrangianIntegrator2 integrator;
  // integrator.set(surface_ls_[SRC].grid().info());
  // integrator.advect(velocity_[SRC], material_, surface_ls_[SRC].grid(),
  //                   surface_ls_[DST].grid(), dt);
}

__global__ void __enforceBoundaries(StaggeredGrid3Accessor vel,
                                    StaggeredGrid3Accessor svel,
                                    RegularGrid3Accessor<MaterialType> m,
                                    bool onlyNormalVelocities) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (m.isIndexStored(i, j, k) && m(i, j, k) == MaterialType::SOLID) {
    if (onlyNormalVelocities) {
      if (m(i + 1, j, k) != MaterialType::SOLID)
        vel.u(i + 1, j, k) = svel.u(i + 1, j, k);
      if (m(i, j + 1, k) != MaterialType::SOLID)
        vel.v(i, j + 1, k) = svel.v(i, j + 1, k);
      if (m(i, j, k + 1) != MaterialType::SOLID)
        vel.w(i, j, k + 1) = svel.v(i, j, k + 1);
      if (m(i - 1, j, k) != MaterialType::SOLID)
        vel.u(i, j, k) = svel.u(i, j, k);
      if (m(i, j - 1, k) != MaterialType::SOLID)
        vel.v(i, j, k) = svel.v(i, j, k);
      if (m(i, j, k - 1) != MaterialType::SOLID)
        vel.w(i, j, k) = svel.w(i, j, k);
    } else {
      vel.u(i, j, k) = svel.u(i, j, k);
      vel.u(i + 1, j, k) = svel.u(i + 1, j, k);
      vel.v(i, j, k) = svel.v(i, j, k);
      vel.v(i, j + 1, k) = svel.v(i, j + 1, k);
      vel.w(i, j, k) = svel.w(i, j, k);
      vel.w(i, j, k + 1) = svel.w(i, j, k + 1);
    }
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::enforceBoundaries(
    size_t buffer, bool onlyNormalVelocities) {
  hermes::ThreadArrayDistributionInfo td(material_.resolution());
  __enforceBoundaries<<<td.gridSize, td.blockSize>>>(
      velocity_[buffer].accessor(), solid_velocity_.accessor(),
      material_.accessor(), onlyNormalVelocities);
}

void propagate(RegularGrid3Hm &mat, RegularGrid3Hf &vel, int dim) {
  MemoryBlock3Hi time_grid(vel.resolution());
  time_grid.allocate();
  auto t = time_grid.accessor();
  auto m = mat.accessor();
  auto velocity = vel.accessor();
  std::queue<vec3i> q;
  for (auto e : t) {
    bool f1 = m(e.i(), e.j(), e.k()) == MaterialType::FLUID;
    bool f2 = m(e.i() - (0 == dim), e.j() - (1 == dim), e.k() - (2 == dim)) ==
              MaterialType::FLUID;
    if (f1 && f2) {
      e.value = 0;
      q.push(vec3i(e.i(), e.j(), e.k()));
    } else
      e.value = Constants::greatest_int();
  }
  while (!q.empty()) {
    vec3i p = q.front();
    q.pop();
    // if (t(p.x, p.y, p.z) > 8)
    //   continue;
    // average velocity from neighbors closer to surface
    float sum = 0;
    int count = 0;
    vec3i dir[6] = {vec3i(-1, 0, 0), vec3i(1, 0, 0),  vec3i(0, -1, 0),
                    vec3i(0, 1, 0),  vec3i(0, 0, -1), vec3i(0, 0, 1)};
    for (int direction = 0; direction < 6; direction++) {
      int i = p.x + dir[direction].x;
      int j = p.y + dir[direction].y;
      int k = p.z + dir[direction].z;
      if (!t.isIndexValid(i, j, k))
        continue;
      if (t(i, j, k) == Constants::greatest_int()) {
        t(i, j, k) = t(p.x, p.y, p.z) + 1;
        q.push(vec3i(i, j, k));
      } else if (t(i, j, k) < t(p.x, p.y, p.z)) {
        sum += velocity(i, j, k);
        count++;
      }
    }
    if (count > 0 && t(p.x, p.y, p.z) > 0)
      velocity(p.x, p.y, p.z) = sum / count;
  }
}

template <>
void PracticalLiquidsSolver3<MemoryLocation::DEVICE>::propagateVelocity() {
  RegularGrid3Hm m(material_);
  StaggeredGrid3H vel;
  vel.copy(velocity_[DST]);
  propagate(m, vel.u(), 0);
  propagate(m, vel.v(), 1);
  propagate(m, vel.w(), 2);
  velocity_[SRC].copy(vel);
}

} // namespace cuda

} // namespace poseidon