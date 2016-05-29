#pragma once

#include "structures/particle_set.h"

#include <functional>
#include <vector>
#include <ponos.h>
using ponos::vec2;
using ponos::BBox2D;

namespace poseidon {

  class ParticleGrid : public ParticleSet<Particle2D> {
  public:
    ParticleGrid();

    void set(uint32_t w, uint32_t h, vec2 offset, vec2 cellSize);

    void fillCell(uint32_t i, uint32_t j);

    void addParticle(uint32_t i, uint32_t j, const Particle2D& p);

    void iterateParticles(BBox2D bbox, std::function<void(const Particle2D& p) > f);

    struct ParticleCell {
      std::vector<uint32_t> particles;
      void add(uint32_t p) {
        particles.emplace_back(p);
      }
    };
    ponos::ZGrid<ParticleCell> grid;
  };

} // poseidon namespace
