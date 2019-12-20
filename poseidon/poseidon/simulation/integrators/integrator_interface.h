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
///\file integrator_interface.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-19
///
///\brief

#ifndef POSEIDON_SIMULATION_INTEGRATOR_INTERFACE_H
#define POSEIDON_SIMULATION_INTEGRATOR_INTERFACE_H

#include "../../numeric/staggered_grid.h"
#include "../definitions.h"

namespace poseidon {

/// Interface class for representing integrators used in simulations.
class Integrator2Interface {
public:
  virtual void setResolution(const ponos::size2 &resolution) {};
  /// Advects a field (input) through the velocity field over a time step. The
  /// advected input field is stored in the output field.
  /// \param velocity velocity field
  /// \param material material field
  /// \param in input field
  /// \param out output field
  /// \param dt time step
  virtual void advect(StaggeredGrid2Accessor velocity,
                      ponos::Grid2Accessor<MaterialType> material,
                      ponos::Grid2Accessor<real_t> in,
                      ponos::Grid2Accessor<real_t> out,
                      real_t dt) = 0;
};

} // poseidon namespace

#endif //POSEIDON_SIMULATION_INTEGRATOR_INTERFACE_H