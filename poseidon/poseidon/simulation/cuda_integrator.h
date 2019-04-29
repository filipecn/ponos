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

#include <hermes/numeric/cuda_staggered_grid.h>

namespace poseidon {

namespace cuda {

class Integrator2 {
public:
  virtual void set(hermes::cuda::Grid2Info info) {}
  virtual void advect(const hermes::cuda::StaggeredGridTexture2 &velocity,
                      const hermes::cuda::GridTexture2<unsigned char> &solid,
                      const hermes::cuda::GridTexture2<float> &phi,
                      hermes::cuda::GridTexture2<float> &phiOut, float dt) = 0;
};

class SemiLagrangianIntegrator2 : public Integrator2 {
public:
  SemiLagrangianIntegrator2();
  void advect(const hermes::cuda::StaggeredGridTexture2 &velocity,
              const hermes::cuda::GridTexture2<unsigned char> &solid,
              const hermes::cuda::GridTexture2<float> &phi,
              hermes::cuda::GridTexture2<float> &phiOut, float dt) override;
};

class MacCormackIntegrator2 : public Integrator2 {
public:
  MacCormackIntegrator2();
  void set(hermes::cuda::Grid2Info info);
  void advect(const hermes::cuda::StaggeredGridTexture2 &velocity,
              const hermes::cuda::GridTexture2<unsigned char> &solid,
              const hermes::cuda::GridTexture2<float> &phi,
              hermes::cuda::GridTexture2<float> &phiOut, float dt) override;

private:
  SemiLagrangianIntegrator2 integrator;
  hermes::cuda::GridTexture2<float> phiNHat, phiN1Hat;
};

} // namespace cuda

} // namespace poseidon

#endif
