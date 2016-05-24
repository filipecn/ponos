#pragma once

#include "structures/particle_grid.h"

#include <ponos.h>
using ponos::vec2;

namespace poseidon {

  class FLIP {
  public:
    void set(uint32_t w, uint32_t h, vec2 offset, float scale);
    // parameters
    float dx;
    float dt;
    float gravity;
    float rho;
    // structures
    ParticleGrid particleGrid;
  };

} // poseidon namespace
