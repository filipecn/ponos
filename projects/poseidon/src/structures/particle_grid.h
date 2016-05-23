#pragma once

#include "structures/particle_set.h"

#include <vector>
#include <ponos.h>
using ponos::Vector2;

namespace poseidon {

  class ParticleGrid : public ParticleSet {
  public:
    ParticleGrid();
    ParticleGrid(int w, int h, Vector2 offset, Vector2 cellSize);

    void set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize);

  private:
    int width, height;
    struct ParticleCell {
      std::vector<int> particles;
    };
    ponos::ZGrid<ParticleCell> grid;
  };

} // poseidon namespace
