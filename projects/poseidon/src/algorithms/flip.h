#pragma once

#include "math/conjugate_gradient.h"
#include "structures/particle_grid.h"

#include <ponos.h>
using ponos::Point2;
using ponos::vec2;
using ponos::ZGrid;

namespace poseidon {

  enum CellType {FLUID = 1, AIR, SOLID, CELLTYPES};

  class FLIP {
  public:
    void set(uint32_t w, uint32_t h, vec2 offset, float scale);
    void fillCell(uint32_t i, uint32_t j);
    void step();
    // parameters
    uint32_t width, height;
    float dx;
    float dt;
    vec2 gravity;
    float rho;
    // structures
    ParticleGrid particleGrid;
    // MAC Grid
    struct VelocityCell {
      float v;
      float wsum;
      VelocityCell() { v = wsum = 0.f; }
    };
    ZGrid<VelocityCell> u, v;
    ZGrid<float> usolid, vsolid;
    ZGrid<char> isSolid;
    ZGrid<char> cell;
    ZGrid<float> vCopy[2];
    // pressure solve
    ConjugateGradient ps;

  // private:
    void gather(ZGrid<VelocityCell>& grid, uint32_t component);
    void addForces(ZGrid<VelocityCell>& grid, uint32_t component);
    void scatter(ZGrid<VelocityCell>& grid, uint32_t component);
    void classifyCells();
    void enforceBoundary();
    void solvePressure();
    void advect();
    ponos::Point2 newPosition(Particle2D pa);
  };

} // poseidon namespace
