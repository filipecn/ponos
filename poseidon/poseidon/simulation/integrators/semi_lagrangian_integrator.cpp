/// Copyright (c) 2019, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file semi_lagrangian_integrator.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-19
///
///\brief

#include <poseidon/simulation/integrators/semi_lagrangian_integrator.h>

namespace poseidon {

SemiLagrangianIntegrator2::SemiLagrangianIntegrator2() = default;

void SemiLagrangianIntegrator2::advect(StaggeredGrid2Accessor velocity,
                                       ponos::Grid2Accessor<MaterialType> material,
                                       ponos::Grid2Accessor<real_t> in,
                                       ponos::Grid2Accessor<real_t> out,
                                       real_t dt) {}

} // poseidon namespace