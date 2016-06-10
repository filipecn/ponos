#pragma once

#include "structures/particle_set.h"

#include <algorithm>
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

    void setPos(uint32_t i, ponos::Point2 p) {
      ponos::Point<int, 2> orig = grid.cell(particles[i].p);
      ponos::Point<int, 2> dest = grid.cell(p);
      particles[i].p = p;
      if (orig != dest) {
        grid(orig[0], orig[1]).remove(i);
        grid(dest[0], dest[1]).add(i);
      }
    }

    void iterateParticles(BBox2D bbox, std::function<void(const Particle2D& p) > f);

    struct ParticleCell {
      int active_num;
      std::vector<int> particles;
      uint32_t numberOfParticles() {
        return active_num;
      }
      void remove(uint32_t p) {
        auto it = std::find(std::begin(particles), std::end(particles), p);
        ASSERT(it != particles.end());
        *it = -1;
        active_num--;
      }
      void add(uint32_t p) {
        auto it = std::find(std::begin(particles), std::end(particles), -1);
        if (it != particles.end())
          *it = p;
        else particles.emplace_back(p);
        active_num++;
      }
      ParticleCell() {
        active_num = 0;
      }
    };
    ponos::ZGrid<ParticleCell> grid;
  };

} // poseidon namespace
