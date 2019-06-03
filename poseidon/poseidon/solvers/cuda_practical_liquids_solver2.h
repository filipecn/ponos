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

#ifndef POSEIDON_SOLVERS_CUDA_PRACTICAL_LIQUIDS_SOLVER2_H
#define POSEIDON_SOLVERS_CUDA_PRACTICAL_LIQUIDS_SOLVER2_H

#include <hermes/numeric/cuda_grid.h>
#include <hermes/parallel/cuda_reduce.h>
#include <poseidon/simulation/cuda_integrator.h>
#include <poseidon/simulation/cuda_level_set.h>

namespace poseidon {

namespace cuda {

enum class MaterialType { FLUID = 0, AIR, SOLID };

using RegularGrid2Dm =
    hermes::cuda::RegularGrid2<hermes::cuda::MemoryLocation::DEVICE,
                               MaterialType>;
using RegularGrid2Hm =
    hermes::cuda::RegularGrid2<hermes::cuda::MemoryLocation::DEVICE,
                               MaterialType>;

// Discretization goes as follows:
// - velocity components are stored on face centers (staggered)
// - level set distances are stored on cell centers
// - all other fields are stored on cell centers
// So, for a NxM resolution simulation, the sizes of the grids are:
// - N+1xM for u velocity component
// - NxM+1 for v velocity component
// - NxM for level set distances
// - NxM for other fields
// For cell i,j:
//              v(i,j+1)
//          --------------
//         |              |
//         |              |
//  u(i,j) |    p(i,j)    | u(i+1,j)
//         |              |
//         |              |
//          --------------
//              v(i,j)
template <hermes::cuda::MemoryLocation L> class PracticalLiquidsSolver2 {
public:
  PracticalLiquidsSolver2() {
    setSpacing(hermes::cuda::vec2f(1.f));
    setOrigin(hermes::cuda::point2f(0.f));
  }
  /// \param res **[in]** resolution in number of cells
  void setResolution(const hermes::cuda::vec2u &res) {
    for (int i = 0; i < 2; i++) {
      velocity_[i].resize(res);
      surface_ls_[i].setResolution(hermes::cuda::vec2u(res.x + 1, res.y + 1));
    }
    material_.resize(res);
    pressure_.resize(res);
    divergence_.resize(res);
  }
  /// \param s **[in]** distance between two cell centers
  void setSpacing(const hermes::cuda::vec2f &s) {
    for (int i = 0; i < 2; i++) {
      velocity_[i].setSpacing(s);
      surface_ls_[i].setSpacing(s);
    }
    material_.setSpacing(s);
    pressure_.setSpacing(s);
    divergence_.setSpacing(s);
  }
  /// \param o **[in]** world position of lowest cell center
  void setOrigin(const hermes::cuda::point2f &o) {
    for (int i = 0; i < 2; i++) {
      velocity_[i].setOrigin(o);
      surface_ls_[i].setOrigin(o);
    }
    material_.setOrigin(o);
    pressure_.setOrigin(o);
    divergence_.setOrigin(o);
  }

private:
  /// \return float maximum value allowed for dt
  float dtFromCFLCondition() {
    float min_s = min(min(material_.spacing().x), material_.spacing().y);
    float max_v = max(hermes::cuda::maxAbs(velocity_[SRC].u()),
                      hermes::cuda::maxAbs(velocity_[SRC].v()));
    if (max_v < 1e-8)
      return hermes::cuda::Constants::greatest<float>();
    return min_s / max_v;
  }

  size_t SRC = 0;
  size_t DST = 1;
  hermes::cuda::StaggeredGrid2<L> velocity_[2];
  LevelSet2<L> surface_ls_[2];
  hermes::cuda::RegularGrid2<L, float> pressure_;
  hermes::cuda::RegularGrid2<L, float> divergence_;
  hermes::cuda::RegularGrid2<L, MaterialType> material_;
};

} // namespace cuda

} // namespace poseidon

#endif