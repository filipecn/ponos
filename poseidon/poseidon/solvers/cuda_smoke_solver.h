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

#ifndef POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H
#define POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H

#include <hermes/hermes.h>
#include <ponos/geometry/point.h>
#include <poseidon/simulation/cuda_integrator.h>
#include <poseidon/simulation/cuda_scene.h>
#include <poseidon/solvers/cuda_smoke_solver_kernels.h>

namespace poseidon {

namespace cuda {

__global__ void __setupScene(poseidon::cuda::Collider2<float> **solids,
                             poseidon::cuda::Collider2<float> **scene) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    solids[0] = new poseidon::cuda::SphereCollider2<float>(
        hermes::cuda::point2(0.f, 0.f), 0.1f);
    float d = 1.0 / 64;
    // floor
    solids[1] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(1.f, d)));
    // ceil
    solids[2] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 1.f - d), hermes::cuda::point2(1.f, 1.f)));
    // left
    solids[3] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(d, 1.f)));
    // right
    solids[4] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(1.f - d, 0.f), hermes::cuda::point2(1.f, 1.f)));
    *scene = new poseidon::cuda::Collider2Set<float>(solids, 5);
  }
}

__global__ void __freeScene(poseidon::cuda::Collider2<float> **solids) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < 5; ++i)
      delete solids[i];
}

__global__ void __rasterColliders(Collider2<float> *const *colliders,
                                  unsigned char *solids, float *u, float *v,
                                  hermes::cuda::Grid2Info sInfo) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * sInfo.resolution.x + x;
  if (x < sInfo.resolution.x && y < sInfo.resolution.y) {
    if ((*colliders)->intersect(sInfo.toWorld(hermes::cuda::point2(x, y))))
      solids[index] = 1;
    else
      solids[index] = 0;
    u[y * sInfo.resolution.x + x] = u[y * sInfo.resolution.x + x + 1] = 0;
    v[y * sInfo.resolution.x + x] = v[(y + 1) * sInfo.resolution.x + x] = 0;
  }
}

/// Eulerian grid based solver for smoke simulations. Stores its data in fast
/// device texture memory.
class GridSmokeSolver2 {
public:
  GridSmokeSolver2() = default;
  ~GridSmokeSolver2() {
    unbindTextures();
    __freeScene<<<1, 1>>>(scene.list);
    using namespace hermes::cuda;
    CUDA_CHECK(cudaFree(scene.list));
    CUDA_CHECK(cudaFree(scene.colliders));
  }
  void setUIntegrator(Integrator2 *integrator) {
    uIntegrator.reset(integrator);
  }
  void setVIntegrator(Integrator2 *integrator) {
    vIntegrator.reset(integrator);
  }
  void setIntegrator(Integrator2 *_integrator) {
    integrator.reset(_integrator);
  }
  void init() {
    setupTextures();
    bindTextures(velocity, density, divergence, pressure, solid, forceField,
                 solidVelocity);
    using namespace hermes::cuda;
    CUDA_CHECK(cudaMalloc(&scene.list, 5 * sizeof(Collider2<float> *)));
    CUDA_CHECK(cudaMalloc(&scene.colliders, sizeof(Collider2<float> *)));
  }
  ///
  /// \param res
  void setResolution(const ponos::uivec2 &res) {
    resolution = hermes::cuda::vec2u(res.x, res.y);
    velocity.resize(resolution);
    density.resize(resolution);
    pressure.resize(resolution);
    divergence.resize(resolution);
    solid.resize(resolution);
    solidVelocity.resize(resolution);
    forceField.resize(resolution);
    integrator->set(density.info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  /// Sets cell size
  /// \param _dx scale
  void setDx(float _dx) {
    dx = _dx;
    velocity.setDx(dx);
    density.setDx(dx);
    pressure.setDx(dx);
    divergence.setDx(dx);
    solid.setDx(dx);
    solidVelocity.setDx(dx);
    forceField.setDx(dx);
    integrator->set(density.info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  /// Sets lower left corner position
  /// \param o offset
  void setOrigin(const ponos::point2f &o) {
    hermes::cuda::point2f p(o.x, o.y);
    velocity.setOrigin(p);
    density.setOrigin(p);
    pressure.setOrigin(p);
    divergence.setOrigin(p);
    solid.setOrigin(p);
    solidVelocity.setOrigin(p);
    forceField.setOrigin(p);
    integrator->set(density.info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  /// Advances one simulation step
  /// \param dt time step
  void step(float dt) {
    using namespace hermes::cuda;
    rasterColliders();
    CUDA_CHECK(cudaDeviceSynchronize());
    uIntegrator->advect(velocity, solid, velocity.u(), velocity.u(), dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    vIntegrator->advect(velocity, solid, velocity.v(), velocity.v(), dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    velocity.u().texture().updateTextureMemory();
    velocity.v().texture().updateTextureMemory();
    integrator->advect(velocity, solid, density, density, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    applyForceField(velocity, forceField, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cerr << "u:\n" << velocity.u().texture() << std::endl;
    // std::cerr << "f:\n" << forceField.v().texture() << std::endl;
    computeDivergence(velocity, solid, divergence);
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cerr << "d:\n" << divergence.texture() << std::endl;
    hermes::cuda::fill(pressure.texture(), 0.0f);
    computePressure(divergence, solid, pressure, dt, 128);
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cerr << "p:\n" << pressure.texture() << std::endl;
    projectionStep(pressure, solid, velocity, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cerr << "v:\n" << velocity.v().texture() << std::endl;
    // reset force field
    // hermes::cuda::fill(forceField.u().texture(), 0.0f);
    // hermes::cuda::fill(forceField.v().texture(), 0.0f);
  }
  /// Raster collider bodies and velocities into grid simulations
  void rasterColliders() {
    __setupScene<<<1, 1>>>(scene.list, scene.colliders);
    hermes::ThreadArrayDistributionInfo td(resolution);
    __rasterColliders<<<td.gridSize, td.blockSize>>>(
        scene.colliders, solid.texture().deviceData(),
        solidVelocity.uDeviceData(), solidVelocity.vDeviceData(), solid.info());
    solid.texture().updateTextureMemory();
    solidVelocity.u().texture().updateTextureMemory();
    solidVelocity.v().texture().updateTextureMemory();
  }
  /// \return hermes::cuda::StaggeredGridTexture2&
  hermes::cuda::StaggeredGridTexture2 &velocityData() { return velocity; }
  /// \return const hermes::cuda::GridTexture2<float>& density data reference
  const hermes::cuda::GridTexture2<float> &pressureData() const {
    return pressure;
  }
  /// \return const hermes::cuda::GridTexture2<float>& density data reference
  const hermes::cuda::GridTexture2<float> &densityData() const {
    return density;
  }
  /// \return hermes::cuda::GridTexture2<float>& density data reference
  hermes::cuda::GridTexture2<float> &densityData() { return density; }
  /// \return const hermes::cuda::GridTexture2<char>&
  const hermes::cuda::GridTexture2<unsigned char> &solidData() const {
    return solid;
  }
  /// \return  hermes::cuda::GridTexture2<char>
  hermes::cuda::GridTexture2<unsigned char> &solidData() { return solid; }
  /// \return hermes::cuda::StaggeredGridTexture2&
  const hermes::cuda::StaggeredGridTexture2 &solidVelocityData() const {
    return solidVelocity;
  }
  /// \return hermes::cuda::StaggeredGridTexture2&
  hermes::cuda::StaggeredGridTexture2 &solidVelocityData() {
    return solidVelocity;
  }
  /// \return hermes::cuda::StaggeredGridTexture2&
  const hermes::cuda::StaggeredGridTexture2 &forceFieldData() const {
    return forceField;
  }
  /// \return hermes::cuda::StaggeredGridTexture2&
  hermes::cuda::StaggeredGridTexture2 &forceFieldData() { return forceField; }

  Scene2<float> scene;

protected:
  std::shared_ptr<Integrator2> vIntegrator;
  std::shared_ptr<Integrator2> uIntegrator;
  std::shared_ptr<Integrator2> integrator;
  hermes::cuda::StaggeredGridTexture2 velocity;
  hermes::cuda::StaggeredGridTexture2 solidVelocity;
  hermes::cuda::StaggeredGridTexture2 forceField;
  hermes::cuda::GridTexture2<float> pressure;
  hermes::cuda::GridTexture2<float> divergence;
  hermes::cuda::GridTexture2<unsigned char> solid;
  hermes::cuda::GridTexture2<float> density;
  hermes::cuda::vec2u resolution;
  float dx = 1;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H