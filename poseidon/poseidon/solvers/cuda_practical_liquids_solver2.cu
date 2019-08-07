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

#include <hermes/numeric/cuda_blas.h>
#include <poseidon/math/cuda_pcg.h>
#include <poseidon/solvers/cuda_practical_liquids_solver2.h>
#include <queue>

namespace poseidon {

namespace cuda {

using namespace hermes::cuda;

__global__ void __computeDistanceNextToSurface(LevelSet2Accessor in,
                                               LevelSet2Accessor out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (out.isIndexStored(i, j)) {
    float minDist = Constants::greatest<float>();
    int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int d = 0; d < 4; d++) {
      int I = i + dir[d][0];
      int J = j + dir[d][1];
      float in_ij = in(i, j);
      float in_IJ = in(I, J);
      if (sign(in_ij) * sign(in_IJ) < 0) {
        float theta = in_ij / (in_ij - in_IJ);
        // if ((i > I || j > J) && in_ij > in_IJ)
        //   theta *= -1;
        float phi = sign(in(i, j)) * theta * in.spacing().x;
        if (fabsf(minDist) > fabs(phi))
          minDist = phi;
      }
    }
    out(i, j) = minDist;
  }
}

__global__ void __handleSolids(LevelSet2Accessor s, LevelSet2Accessor f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (f.isIndexStored(i, j)) {
    if (f(i, j) < 0 && s(i, j) < 0) {
      f(i, j) = -s(i, j);
    }
    // else if (f(i, j) < 0 && fabs(f(i, j)) > fabs(s(i, j))) {
    //   f(i, j) = sign(f(i, j)) * fabs(s(i, j));
    // }
  }
}

void propagate(LevelSet2H &ls, int s) {
  auto phi = ls.grid().accessor();
  MemoryBlock2Huc frozenGrid(phi.resolution());
  frozenGrid.allocate();
  auto frozen = frozenGrid.accessor();
  fill2(frozenGrid, (unsigned char)0);
  struct Point_ {
    Point_(int I, int J, float D) : i(I), j(J), t(D) {}
    int i, j;
    float t; // tentative distance
  };
  auto cmp = [](Point_ a, Point_ b) { return a.t > b.t; };
  std::priority_queue<Point_, std::vector<Point_>, decltype(cmp)> q(cmp);
  // get initial set of points
  for (auto e : phi)
    if (fabsf(e.value) < Constants::greatest<float>() && sign(e.value) * s > 0)
      q.push(Point_(e.i(), e.j(), fabsf(e.value)));
  while (!q.empty()) {
    Point_ p = q.top();
    q.pop();
    frozen(p.i, p.j) = 2;
    phi(p.i, p.j) = s * p.t;
    vec2i dir[4] = {vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1)};
    for (int direction = 0; direction < 4; direction++) {
      int i = p.i + dir[direction].x;
      int j = p.j + dir[direction].y;
      if (!frozen.isIndexValid(i, j) || frozen(i, j) ||
          (fabs(phi(i, j)) < Constants::greatest<float>() &&
           sign(phi(i, j)) * s < 0))
        continue;
      float phi1 = fminf(fabsf(phi(p.i - 1, p.j)), fabsf(phi(p.i + 1, p.j)));
      float phi2 = fminf(fabsf(phi(p.i, p.j - 1)), fabsf(phi(p.i, p.j + 1)));
      if (phi2 < phi1)
        swap(phi1, phi2);
      // try closest neighbor
      float d = phi1 + phi.spacing().x; // TODO: assuming square cells
      if (d > phi2)
        d = (phi1 + phi2 +
             sqrtf(2.f * phi.spacing().x * phi.spacing().x -
                   (phi2 - phi1) * (phi2 - phi1))) *
            0.5f;
      phi(i, j) = s * fminf(fabsf(phi(i, j)), d);
      q.push(Point_(i, j, fabsf(phi(i, j))));
      frozen(i, j) = 1;
    }
  }
}

template <>
void PracticalLiquidsSolver2<
    hermes::cuda::MemoryLocation::DEVICE>::updateSurfaceDistanceField() {
  hermes::ThreadArrayDistributionInfo td(surface_ls_[DST].grid().resolution());
  // initiate all domain with infinity distance
  fill2(surface_ls_[DST].grid().data(), Constants::greatest<float>());
  __handleSolids<<<td.gridSize, td.blockSize>>>(solid_ls_.accessor(),
                                                surface_ls_[SRC].accessor());
  // compute distances of points closest to surface
  __computeDistanceNextToSurface<<<td.gridSize, td.blockSize>>>(
      surface_ls_[SRC].accessor(), surface_ls_[DST].accessor());
  {
    LevelSet2H hls(surface_ls_[DST]);
    // propagate distances in positive direction
    propagate(hls, 1);
    // propagate distances in negative direction
    propagate(hls, -1);
    memcpy(surface_ls_[DST].grid().data(), hls.grid().data());
  }
  // std::cerr << solid_ls_.grid().data() << std::endl;
  // std::cerr << surface_ls_[DST].grid().data() << std::endl;
  __handleSolids<<<td.gridSize, td.blockSize>>>(solid_ls_.accessor(),
                                                surface_ls_[DST].accessor());
  // std::cerr << surface_ls_[DST].grid().data() << std::endl;
  // exit(0);
}

__global__ void __classifyCells(LevelSet2Accessor ls, LevelSet2Accessor sls,
                                RegularGrid2Accessor<MaterialType> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (m.isIndexStored(i, j)) {
    if (sls(i, j) <= 0)
      m(i, j) = MaterialType::SOLID;
    else if (ls(i, j) <= 0)
      m(i, j) = MaterialType::FLUID;
    else
      m(i, j) = MaterialType::AIR;
  }
}

template <>
void PracticalLiquidsSolver2<
    hermes::cuda::MemoryLocation::DEVICE>::classifyCells() {
  hermes::ThreadArrayDistributionInfo td(material_.resolution());
  __classifyCells<<<td.gridSize, td.blockSize>>>(
      surface_ls_[DST].accessor(), solid_ls_.accessor(), material_.accessor());
}

__global__ void __rasterSolid(LevelSet2Accessor ls, StaggeredGrid2Accessor vel,
                              bbox2f box, vec2f v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (ls.isIndexStored(i, j)) {
    ls(i, j) = fminf(ls(i, j), SDF::box(box, ls.worldPosition(i, j)));
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::rasterSolid(
    const bbox2f &box, const vec2f &velocity) {
  hermes::ThreadArrayDistributionInfo td(solid_ls_.grid().resolution());
  __rasterSolid<<<td.gridSize, td.blockSize>>>(
      solid_ls_.accessor(), solid_velocity_.accessor(), box, velocity);
}

__global__ void __applyExternalForces(RegularGrid2Accessor<float> in,
                                      RegularGrid2Accessor<float> out,
                                      VectorGrid2Accessor force,
                                      RegularGrid2Accessor<MaterialType> m,
                                      float dt, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (in.isIndexStored(i, j)) {
    if (m(i - (0 | !dim), j - (1 & dim)) == MaterialType::FLUID &&
        m(i, j) == MaterialType::FLUID) {
      float f =
          (force(i - (0 | !dim), j - (1 & dim))[dim] + force(i, j)[dim]) * 0.5f;
      out(i, j) = in(i, j) + dt * f;
    }
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::applyExternalForces(
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
}

__global__ void __computeDivergence(StaggeredGrid2Accessor vel,
                                    RegularGrid2Accessor<MaterialType> solid,
                                    RegularGrid2Accessor<float> divergence,
                                    vec2f invdx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (divergence.isIndexStored(i, j)) {
    float left = vel.u(i, j);
    float right = vel.u(i + 1, j);
    float bottom = vel.v(i, j);
    float top = vel.v(i, j + 1);
    bool sleft = solid(i - 1, j) == MaterialType::SOLID;
    bool sright = solid(i + 1, j) == MaterialType::SOLID;
    bool sbottom = solid(i, j - 1) == MaterialType::SOLID;
    bool stop = solid(i, j + 1) == MaterialType::SOLID;
    if (sleft)
      left = 0; // tex3D(uSolidTex3, xc, yc, zc);
    if (sright)
      right = 0; // tex3D(uSolidTex3, xc + 1, yc, zc);
    if (sbottom)
      bottom = 0; // tex3D(vSolidTex3, xc, yc, zc);
    if (stop)
      top = 0; // tex3D(vSolidTex3, xc, yc + 1, zc);
    divergence(i, j) = dot(invdx, vec2f(right - left, top - bottom));
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::computeDivergence() {
  vec2f inv(-1.f / divergence_.spacing().x);
  hermes::ThreadArrayDistributionInfo td(divergence_.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      velocity_[DST].accessor(), material_.accessor(), divergence_.accessor(),
      inv);
}

__global__ void __fillPressureMatrix(MemoryBlock2Accessor<FDMatrix2Entry> A,
                                     RegularGrid2Accessor<MaterialType> m,
                                     float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (A.isIndexValid(i, j)) {
    if (m(i, j) == MaterialType::SOLID)
      return;
    A(i, j).diag = 0;
    A(i, j).x = 0;
    A(i, j).y = 0;
    if (m.isIndexStored(i - 1, j) && m(i - 1, j) != MaterialType::SOLID)
      A(i, j).diag += scale;
    if (m.isIndexStored(i + 1, j) && m(i + 1, j) == MaterialType::FLUID) {
      A(i, j).diag += scale;
      A(i, j).x = -scale;
    } else if (m.isIndexStored(i + 1, j) && m(i + 1, j) == MaterialType::AIR)
      A(i, j).diag += scale;
    if (m.isIndexStored(i, j - 1) && m(i, j - 1) != MaterialType::SOLID)
      A(i, j).diag += scale;
    if (m.isIndexStored(i, j + 1) && m(i, j + 1) == MaterialType::FLUID) {
      A(i, j).diag += scale;
      A(i, j).y = -scale;
    } else if (m.isIndexStored(i, j + 1) && m(i, j + 1) == MaterialType::AIR)
      A(i, j).diag += scale;
  }
}

__global__ void __buildRHS(MemoryBlock2Accessor<int> indices,
                           RegularGrid2Accessor<float> divergence,
                           MemoryBlock1Accessor<double> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      rhs[indices(i, j)] = divergence(i, j);
  }
}

__global__ void __1To2(MemoryBlock2Accessor<int> indices,
                       MemoryBlock1Accessor<double> v,
                       MemoryBlock2Accessor<float> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      m(i, j) = v[indices(i, j)];
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::solvePressure(float dt) {
  // fill matrix
  float scale = dt / (divergence_.spacing().x * divergence_.spacing().x);
  {
    hermes::ThreadArrayDistributionInfo td(divergence_.resolution());
    __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
        pressure_matrix_.dataAccessor(), material_.accessor(), scale);
  }
  // compute indices
  auto res = divergence_.resolution();
  MemoryBlock2Hi h_indices(res);
  h_indices.allocate();
  MemoryBlock2<MemoryLocation::HOST, MaterialType> h_material(res);
  h_material.allocate();
  memcpy(h_material, material_.data());
  auto m = h_material.accessor();
  int curIndex = 0;
  for (auto e : h_indices.accessor())
    if (m(e.i(), e.j()) == MaterialType::FLUID)
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
    FDMatrix2H H(pressure_matrix_.gridSize());
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
    __1To2<<<td.gridSize, td.blockSize>>>(pressure_matrix_.indexDataAccessor(),
                                          x.accessor(),
                                          pressure_.data().accessor());
  }
}

__global__ void __projectionStep(RegularGrid2Accessor<float> vel,
                                 RegularGrid2Accessor<float> pressure,
                                 RegularGrid2Accessor<MaterialType> m, vec2u d,
                                 float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (vel.isIndexStored(i, j)) {
    if (m(i - d.x, j - d.y) == MaterialType::SOLID)
      vel(i, j) = 0; // tex3D(wSolidTex3, xc, yc, zc - 1);
    else if (m(i, j) == MaterialType::SOLID)
      vel(i, j) = 0; // tex3D(wSolidTex3, xc, yc, zc);
    else {
      float b = pressure(i - d.x, j - d.y);
      float f = pressure(i, j);
      vel(i, j) -= scale * (f - b);
    }
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::project(float dt) {
  {
    auto info = velocity_[DST].u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity_[DST].u().accessor(), pressure_.accessor(),
        material_.accessor(), vec2u(1, 0), scale);
  }
  {
    auto info = velocity_[DST].v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity_[DST].v().accessor(), pressure_.accessor(),
        material_.accessor(), vec2u(0, 1), scale);
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::advectVelocity(float dt) {
  SemiLagrangianIntegrator2 integrator;
  integrator.advect(velocity_[SRC], material_, velocity_[SRC].u(),
                    velocity_[DST].u(), dt);
  integrator.advect(velocity_[SRC], material_, velocity_[SRC].v(),
                    velocity_[DST].v(), dt);
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::advectFluid(float dt) {
  fluidIntegrator.advect(velocity_[SRC], material_, surface_ls_[SRC].grid(),
                         surface_ls_[DST].grid(), dt);
}

__global__ void __enforceBoundaries(StaggeredGrid2Accessor vel,
                                    StaggeredGrid2Accessor svel,
                                    RegularGrid2Accessor<MaterialType> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (m.isIndexStored(i, j) && m(i, j) == MaterialType::SOLID) {
    vel.u(i, j) = svel.u(i, j);
    vel.u(i + 1, j) = svel.u(i + 1, j);
    vel.v(i, j) = svel.v(i, j);
    vel.v(i, j + 1) = svel.v(i, j + 1);
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::enforceBoundaries() {
  hermes::ThreadArrayDistributionInfo td(material_.resolution());
  __enforceBoundaries<<<td.gridSize, td.blockSize>>>(velocity_[DST].accessor(),
                                                     solid_velocity_.accessor(),
                                                     material_.accessor());
}

void propagate(RegularGrid2Hm &mat, RegularGrid2Hf &vel, int dim) {
  MemoryBlock2Hi time_grid(vel.resolution());
  time_grid.allocate();
  auto t = time_grid.accessor();
  auto m = mat.accessor();
  auto velocity = vel.accessor();
  std::queue<vec2i> q;
  auto isCellSurface = [&](int i, int j) -> bool {
    return m(i, j) == MaterialType::FLUID &&
           (m(i - 1, j) != m(i, j) || m(i + 1, j) != m(i, j) ||
            m(i, j - 1) != m(i, j) || m(i, j + 1) != m(i, j));
  };
  for (auto e : t) {
    bool f1 = m(e.i(), e.j()) == MaterialType::FLUID;
    bool f2 = m(e.i() - (0 | !dim), e.j() - (1 & dim)) == MaterialType::FLUID;
    if (f1 && f2) {
      e.value = 0;
      // if (isCellSurface(e.i(), e.j()) ||
      //     isCellSurface(e.i() - (0 | !dim), e.j() - (1 & dim))) {
      //   e.value = 0;
      q.push(vec2i(e.i(), e.j()));
      // }
    } else
      e.value = Constants::greatest_int();
  }
  while (!q.empty()) {
    vec2i p = q.front();
    q.pop();
    if (t(p.x, p.y) > 8)
      continue;
    // average velocity from neighbors closer to surface
    float sum = 0;
    int count = 0;
    vec2i dir[4] = {vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1)};
    for (int direction = 0; direction < 4; direction++) {
      int i = p.x + dir[direction].x;
      int j = p.y + dir[direction].y;
      if (!t.isIndexValid(i, j))
        continue;
      if (t(i, j) == Constants::greatest_int()) {
        t(i, j) = t(p.x, p.y) + 1;
        q.push(vec2i(i, j));
      } else if (t(i, j) < t(p.x, p.y)) {
        sum += velocity(i, j);
        count++;
      }
    }
    if (count > 0 && t(p.x, p.y) > 0)
      velocity(p.x, p.y) = sum / count;
  }
}

template <>
void PracticalLiquidsSolver2<MemoryLocation::DEVICE>::propagateVelocity() {
  RegularGrid2Hm m(material_);
  StaggeredGrid2H vel;
  vel.copy(velocity_[DST]);
  propagate(m, vel.u(), 0);
  propagate(m, vel.v(), 1);
  velocity_[SRC].copy(vel);
}

__global__ void __computeKineticEnergy(StaggeredGrid2Accessor data,
                                       RegularGrid2Accessor<MaterialType> m,
                                       float *c) {
  __shared__ float cache[256];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheindex = threadIdx.x;
  int n = data.resolution.x * data.resolution.y;

  float temp = 0.0;
  while (tid < n) {
    int i = tid / data.resolution.x;
    int j = tid % data.resolution.x;
    if (m(i, j) == MaterialType::FLUID)
      temp += data(i, j).length();
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheindex] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheindex < i)
      cache[cacheindex] += cache[cacheindex + i];
    __syncthreads();
    i /= 2;
  }
  if (cacheindex == 0)
    c[blockIdx.x] = cache[0];
}

template <>
void PracticalLiquidsSolver2<
    MemoryLocation::DEVICE>::computeFluidKineticEnergy() {
  auto data = velocity_[DST].accessor();
  size_t blockSize = (data.resolution.x * data.resolution.y + 256 - 1) / 256;
  if (blockSize > 32)
    blockSize = 32;
  float *c = new float[blockSize];
  float *d_c;
  cudaMalloc((void **)&d_c, blockSize * sizeof(float));
  __computeKineticEnergy<<<blockSize, 256>>>(data, material_.accessor(), d_c);
  cudaMemcpy(c, d_c, blockSize * sizeof(float), cudaMemcpyDeviceToHost);
  float sum = 0;
  for (int i = 0; i < blockSize; i++)
    sum += c[i];
  cudaFree(d_c);
  delete[] c;
  std::cerr << "kin " << sum << " ";
}

} // namespace cuda

} // namespace poseidon