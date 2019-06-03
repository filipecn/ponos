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

#ifndef POSEIDON_SIMULATION_CUDA_INTEGRATOR_H
#define POSEIDON_SIMULATION_CUDA_INTEGRATOR_H

#include <hermes/common/defs.h>
#include <hermes/numeric/cuda_staggered_grid.h>

namespace poseidon {

namespace cuda {

class Integrator2 {
public:
  virtual void set(hermes::cuda::Grid2Info info) {}
  virtual void set(hermes::cuda::RegularGrid2Info info) {}
  // TODO: DEPRECATED
  virtual void advect(hermes::cuda::VectorGridTexture2 &velocity,
                      hermes::cuda::GridTexture2<unsigned char> &solid,
                      hermes::cuda::GridTexture2<float> &phi,
                      hermes::cuda::GridTexture2<float> &phiOut, float dt) = 0;
  virtual void advect(hermes::cuda::StaggeredGrid2D &velocity,
                      hermes::cuda::RegularGrid2Duc &solid,
                      hermes::cuda::RegularGrid2Df &solidPhi,
                      hermes::cuda::RegularGrid2Df &phi,
                      hermes::cuda::RegularGrid2Df &phiOut, float dt) = 0;
  virtual void advect_t(hermes::cuda::VectorGrid2D &velocity,
                        hermes::cuda::RegularGrid2Duc &solid,
                        hermes::cuda::RegularGrid2Df &phi,
                        hermes::cuda::RegularGrid2Df &phiOut, float dt) = 0;
};

class Integrator3 {
public:
  virtual void set(hermes::cuda::RegularGrid3Info info) {}
  virtual void advect(hermes::cuda::StaggeredGrid3D &velocity,
                      hermes::cuda::RegularGrid3Duc &solid,
                      hermes::cuda::RegularGrid3Df &phi,
                      hermes::cuda::RegularGrid3Df &phiOut, float dt) = 0;
  virtual void advect_t(hermes::cuda::VectorGrid3D &velocity,
                        hermes::cuda::RegularGrid3Duc &solid,
                        hermes::cuda::RegularGrid3Df &phi,
                        hermes::cuda::RegularGrid3Df &phiOut, float dt) = 0;
};

class SemiLagrangianIntegrator2 : public Integrator2 {
public:
  SemiLagrangianIntegrator2();
  // TODO: DEPRECATED
  void advect(hermes::cuda::VectorGridTexture2 &velocity,
              hermes::cuda::GridTexture2<unsigned char> &solid,
              hermes::cuda::GridTexture2<float> &phi,
              hermes::cuda::GridTexture2<float> &phiOut, float dt) override;
  void advect(hermes::cuda::StaggeredGrid2D &velocity,
              hermes::cuda::RegularGrid2Duc &solid,
              hermes::cuda::RegularGrid2Df &solidPhi,
              hermes::cuda::RegularGrid2Df &phi,
              hermes::cuda::RegularGrid2Df &phiOut, float dt) override;
  void advect_t(hermes::cuda::VectorGrid2D &velocity,
                hermes::cuda::RegularGrid2Duc &solid,
                hermes::cuda::RegularGrid2Df &phi,
                hermes::cuda::RegularGrid2Df &phiOut, float dt) override;
};

class SemiLagrangianIntegrator3 : public Integrator3 {
public:
  SemiLagrangianIntegrator3();
  void advect(hermes::cuda::StaggeredGrid3D &velocity,
              hermes::cuda::RegularGrid3Duc &solid,
              hermes::cuda::RegularGrid3Df &phi,
              hermes::cuda::RegularGrid3Df &phiOut, float dt) override;
  void advect_t(hermes::cuda::VectorGrid3D &velocity,
                hermes::cuda::RegularGrid3Duc &solid,
                hermes::cuda::RegularGrid3Df &phi,
                hermes::cuda::RegularGrid3Df &phiOut, float dt) override;
};

class MacCormackIntegrator2 : public Integrator2 {
public:
  MacCormackIntegrator2();
  void set(hermes::cuda::Grid2Info info);
  void set(hermes::cuda::RegularGrid2Info info);
  // TODO: DEPRECATED
  void advect(hermes::cuda::VectorGridTexture2 &velocity,
              hermes::cuda::GridTexture2<unsigned char> &solid,
              hermes::cuda::GridTexture2<float> &phi,
              hermes::cuda::GridTexture2<float> &phiOut, float dt) override;
  void advect(hermes::cuda::StaggeredGrid2D &velocity,
              hermes::cuda::RegularGrid2Duc &solid,
              hermes::cuda::RegularGrid2Df &solidPhi,
              hermes::cuda::RegularGrid2Df &phi,
              hermes::cuda::RegularGrid2Df &phiOut, float dt) override;
  void advect_t(hermes::cuda::VectorGrid2D &velocity,
                hermes::cuda::RegularGrid2Duc &solid,
                hermes::cuda::RegularGrid2Df &phi,
                hermes::cuda::RegularGrid2Df &phiOut, float dt) override;

private:
  SemiLagrangianIntegrator2 integrator;
  hermes::cuda::GridTexture2<float> phiNHat_t, phiN1Hat_t;
  hermes::cuda::RegularGrid2Df phiNHat, phiN1Hat;
};

class MacCormackIntegrator3 : public Integrator3 {
public:
  MacCormackIntegrator3();
  void set(hermes::cuda::RegularGrid3Info info);
  void advect(hermes::cuda::StaggeredGrid3D &velocity,
              hermes::cuda::RegularGrid3Duc &solid,
              hermes::cuda::RegularGrid3Df &phi,
              hermes::cuda::RegularGrid3Df &phiOut, float dt) override;
  void advect_t(hermes::cuda::VectorGrid3D &velocity,
                hermes::cuda::RegularGrid3Duc &solid,
                hermes::cuda::RegularGrid3Df &phi,
                hermes::cuda::RegularGrid3Df &phiOut, float dt) override;

private:
  SemiLagrangianIntegrator3 integrator;
  hermes::cuda::RegularGrid3Df phiNHat, phiN1Hat;
};

/// Implements Hamilton-Jacobi essentially nonoscillatory (ENO) polynomial
/// interpolation for advecting fields. The polynomial is also constructed based
/// on upwind differencing.
class ENOIntegrator2 {
public:
  void set(hermes::cuda::RegularGrid2Info info);
  void advect(hermes::cuda::StaggeredGrid2D &velocity,
              hermes::cuda::RegularGrid2Duc &solid,
              hermes::cuda::RegularGrid2Df &solidPhi,
              hermes::cuda::RegularGrid2Df &phi,
              hermes::cuda::RegularGrid2Df &phiOut, float dt);

private:
  size_t order = 3;
};

} // namespace cuda

} // namespace poseidon

#endif
