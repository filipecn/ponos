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

#include <ponos/geometry/point.h>
#include <poseidon/simulation/cuda_integrator.h>
#include <poseidon/simulation/cuda_scene.h>
#include <poseidon/solvers/cuda_smoke_solver_kernels.h>

namespace poseidon {

namespace cuda {

__global__ void __setupScene(poseidon::cuda::Collider2<float> **solids,
                             poseidon::cuda::Collider2<float> **scene) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float d = 1.0 / 64;
    // floor
    solids[0] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(1.f, d)));
    // ceil
    solids[1] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 1.f - d), hermes::cuda::point2(1.f, 1.f)));
    // left
    solids[2] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(0.f, 0.f), hermes::cuda::point2(d, 1.f)));
    // right
    solids[3] = new poseidon::cuda::BoxCollider2<float>(hermes::cuda::bbox2(
        hermes::cuda::point2(1.f - d, 0.f), hermes::cuda::point2(1.f, 1.f)));
    solids[4] = new poseidon::cuda::SphereCollider2<float>(
        hermes::cuda::point2(0.5f, 0.5f), 0.1f);
    *scene = new poseidon::cuda::Collider2Set<float>(solids, 4);
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

__global__ void __normalizeIFFT(float *g_data, int width, int height, float N) {

  // index = x * height + y

  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  int index = yIndex * width + xIndex;

  g_data[index] = g_data[index] / N;
}

/// Eulerian grid based solver for smoke simulations. Stores its data in fast
/// device texture memory.
template <typename GridType> class GridSmokeSolver2 {
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
    bindTextures(velocity, velocityCopy, scalarFields[0], divergence, pressure,
                 solid, forceField, solidVelocity);
    using namespace hermes::cuda;
    CUDA_CHECK(cudaMalloc(&scene.list, 5 * sizeof(Collider2<float> *)));
    CUDA_CHECK(cudaMalloc(&scene.colliders, sizeof(Collider2<float> *)));
    // create fft plans
    auto uRes = velocity.u().resolution();
    auto vRes = velocity.v().resolution();
    if (cufftPlan2d(&forwardPlanU, uRes.x, uRes.y, CUFFT_R2C) !=
        CUFFT_SUCCESS) {
      std::cerr << "CUFFT Error: Failed to create plan\n";
      exit(-1);
    }
    if (cufftPlan2d(&inversePlanU, uRes.x, uRes.y, CUFFT_C2R) !=
        CUFFT_SUCCESS) {
      std::cerr << "CUFFT Error: Failed to create plan\n";
      exit(-1);
    }
    if (cufftPlan2d(&forwardPlanV, vRes.x, vRes.y, CUFFT_R2C) !=
        CUFFT_SUCCESS) {
      std::cerr << "CUFFT Error: Failed to create plan\n";
      exit(-1);
    }
    if (cufftPlan2d(&inversePlanV, vRes.x, vRes.y, CUFFT_C2R) !=
        CUFFT_SUCCESS) {
      std::cerr << "CUFFT Error: Failed to create plan\n";
      exit(-1);
    }
    // allocate device memory for frequency space
    CUDA_CHECK(cudaMalloc((void **)&d_frequenciesU,
                          sizeof(cufftComplex) * uRes.x * (uRes.y / 2 + 1)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_frequenciesV,
                   sizeof(cufftComplex) * vRes.x * ((vRes.y + 1) / 2 + 1)));
  }
  ///
  /// \param res
  void setResolution(const ponos::uivec2 &res) {
    resolution = hermes::cuda::vec2u(res.x, res.y);
    velocity.resize(resolution);
    velocityCopy.resize(resolution);
    for (auto &f : scalarFields)
      f.resize(resolution);
    pressure.resize(resolution);
    divergence.resize(resolution);
    solid.resize(resolution);
    solidVelocity.resize(resolution);
    forceField.resize(resolution);
    if (scalarFields.size())
      integrator->set(scalarFields[0].info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  /// Sets cell size
  /// \param _dx scale
  void setDx(float _dx) {
    dx = _dx;
    velocity.setDx(dx);
    velocityCopy.setDx(dx);
    for (auto &f : scalarFields)
      f.setDx(dx);
    pressure.setDx(dx);
    divergence.setDx(dx);
    solid.setDx(dx);
    solidVelocity.setDx(dx);
    forceField.setDx(dx);
    if (scalarFields.size())
      integrator->set(scalarFields[0].info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  /// Sets lower left corner position
  /// \param o offset
  void setOrigin(const ponos::point2f &o) {
    hermes::cuda::point2f p(o.x, o.y);
    velocity.setOrigin(p);
    velocityCopy.setOrigin(p);
    for (auto &f : scalarFields)
      f.setOrigin(p);
    pressure.setOrigin(p);
    divergence.setOrigin(p);
    solid.setOrigin(p);
    solidVelocity.setOrigin(p);
    forceField.setOrigin(p);
    if (scalarFields.size())
      integrator->set(scalarFields[0].info());
    uIntegrator->set(velocity.u().info());
    vIntegrator->set(velocity.v().info());
  }
  size_t addScalarField() {
    scalarFields.emplace_back();
    return scalarFields.size() - 1;
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
    for (size_t i = 0; i < scalarFields.size(); i++)
      integrator->advect(velocity, solid, scalarFields[i], scalarFields[i], dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    applyForceField(velocity, forceField, dt);
    // velocityCopy.copy(velocity);
    // velocityCopy.u().texture().updateTextureMemory();
    // velocityCopy.v().texture().updateTextureMemory();
    // diffuse(velocity, 0.4, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    computeDivergence(velocity, solid, divergence);
    CUDA_CHECK(cudaDeviceSynchronize());
    computePressure(divergence, solid, pressure, dt, 128);
    CUDA_CHECK(cudaDeviceSynchronize());
    projectionStep(pressure, solid, velocity, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    hermes::cuda::fill(forceField.u().texture(), 0.0f);
    hermes::cuda::fill(forceField.v().texture(), 0.0f);
  }
  /// Advances one simulation step folllowing original Stan method
  /// \param dt time step
  void stepFFT(float dt) {
    using namespace hermes::cuda;
    applyForceField(velocity, forceField, dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    uIntegrator->advect(velocity, solid, velocity.u(), velocity.u(), dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    vIntegrator->advect(velocity, solid, velocity.v(), velocity.v(), dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    velocity.u().texture().updateTextureMemory();
    velocity.v().texture().updateTextureMemory();
    for (size_t i = 0; i < scalarFields.size(); i++)
      integrator->advect(velocity, solid, scalarFields[i], scalarFields[i], dt);
    CUDA_CHECK(cudaDeviceSynchronize());
    applyFFT();
    diffuseFFT(resolution, d_frequenciesU, d_frequenciesV, 0.01, dt);
    projectFFT(resolution, d_frequenciesU, d_frequenciesV);
    applyiFFT();
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
  GridType &velocityData() { return velocity; }
  /// \return const hermes::cuda::GridTexture2<float>& density data reference
  const hermes::cuda::GridTexture2<float> &pressureData() const {
    return pressure;
  }
  const hermes::cuda::GridTexture2<float> &scalarField(size_t i) const {
    return scalarFields[i];
  }
  hermes::cuda::GridTexture2<float> &scalarField(size_t i) {
    return scalarFields[i];
  }
  /// \return const hermes::cuda::GridTexture2<char>&
  const hermes::cuda::GridTexture2<unsigned char> &solidData() const {
    return solid;
  }
  /// \return  hermes::cuda::GridTexture2<char>
  hermes::cuda::GridTexture2<unsigned char> &solidData() { return solid; }
  /// \return hermes::cuda::StaggeredGridTexture2&
  const GridType &solidVelocityData() const { return solidVelocity; }
  /// \return hermes::cuda::StaggeredGridTexture2&
  GridType &solidVelocityData() { return solidVelocity; }
  /// \return hermes::cuda::StaggeredGridTexture2&
  const GridType &forceFieldData() const { return forceField; }
  /// \return hermes::cuda::StaggeredGridTexture2&
  GridType &forceFieldData() { return forceField; }

  Scene2<float> scene;

protected:
  void applyFFT() {
    using namespace hermes::cuda;
    if (cufftExecR2C(forwardPlanU, velocity.uDeviceData(), d_frequenciesU) !=
        CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
      exit(-1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cufftExecR2C(forwardPlanV, velocity.vDeviceData(), d_frequenciesV) !=
        CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
      exit(-1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void applyiFFT() {
    using namespace hermes::cuda;
    if (cufftExecC2R(inversePlanU, d_frequenciesU, velocity.uDeviceData()) !=
        CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
      exit(-1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    {
      dim3 grid(resolution.x / 16, resolution.y / 16, 1);
      dim3 threads(16, 16, 1);
      __normalizeIFFT<<<grid, threads>>>(velocity.uDeviceData(), resolution.x,
                                         resolution.y,
                                         resolution.x * resolution.y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cufftExecC2R(inversePlanV, d_frequenciesV, velocity.vDeviceData()) !=
        CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
      exit(-1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    {
      dim3 grid(resolution.x / 16, resolution.y / 16, 1);
      dim3 threads(16, 16, 1);
      __normalizeIFFT<<<grid, threads>>>(velocity.vDeviceData(), resolution.x,
                                         resolution.y,
                                         resolution.x * resolution.y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::shared_ptr<Integrator2> vIntegrator;
  std::shared_ptr<Integrator2> uIntegrator;
  std::shared_ptr<Integrator2> integrator;
  GridType velocity, velocityCopy;
  GridType solidVelocity;
  GridType forceField;
  hermes::cuda::GridTexture2<float> pressure;
  hermes::cuda::GridTexture2<float> divergence;
  hermes::cuda::GridTexture2<unsigned char> solid;
  std::vector<hermes::cuda::GridTexture2<float>> scalarFields;
  hermes::cuda::vec2u resolution;
  float dx = 1;
  // fft
  cufftComplex *d_frequenciesU, *d_frequenciesV;
  cufftHandle forwardPlanU, inversePlanU;
  cufftHandle forwardPlanV, inversePlanV;
};

} // namespace cuda

} // namespace poseidon

#endif // POSEIDON_SOLVERS_CUDA_SMOKE_SOLVER_H