#pragma once

#include "structures/particle_grid.h"

#include <ponos.h>
using ponos::Point2;
using ponos::vec2;
using ponos::ZGrid;

namespace poseidon {

  class FLIP {
  public:
    void set(uint32_t w, uint32_t h, vec2 offset, float scale);
    void fillCell(uint32_t i, uint32_t j);
    void step();
    // parameters
    float dx;
    float dt;
    float gravity;
    float rho;
    // structures
    ParticleGrid particleGrid;
    // MAC Grid
    ZGrid<float> u, v, p;
    ZGrid<float> vCopy[2];

  private:
    void gather(ZGrid<float>& grid);
  };

} // poseidon namespace
