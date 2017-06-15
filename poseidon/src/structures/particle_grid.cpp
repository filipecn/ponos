#include "structures/particle_grid.h"

namespace poseidon {
ParticleGrid::ParticleGrid() {}

void ParticleGrid::set(uint32_t w, uint32_t h, vec2 offset, vec2 cellSize) {
  grid.setDimensions(w, h);
  grid.setTransform(offset, cellSize);
}

void ParticleGrid::fillCell(uint32_t i, uint32_t j) {
  float delta[] = {-0.25f, 0.25f};
  for (int k = 0; k < 2; ++k)
    for (int kk = 0; kk < 2; ++kk) {
      Particle2D p(
          grid.toWorld(ponos::Point2(static_cast<float>(i) + delta[k],
                                     static_cast<float>(j) + delta[kk])),
          ponos::Vector2());
      addParticle(i, j, p);
    }
}

void ParticleGrid::addParticle(uint32_t i, uint32_t j, const Particle2D &p) {
  grid(i, j).add(particles.size());
  particles.emplace_back(p);
}

void ParticleGrid::iterateParticles(
    BBox2D bbox, std::function<void(const Particle2D &p)> f) {
  ponos::Point2 p1 = grid.toGrid(bbox.pMin);
  ponos::Point2 p2 = grid.toGrid(bbox.pMax);
  int xmin = std::max(0, static_cast<int>(p1.x));
  int xmax =
      std::min(static_cast<int>(p2.x + 0.5f), static_cast<int>(grid.width - 1));
  int ymin = std::max(0, static_cast<int>(p1.y));
  int ymax = std::min(static_cast<int>(p2.y + 0.5f),
                      static_cast<int>(grid.height - 1));
  for (int i = xmin; i <= xmax; ++i)
    for (int j = ymin; j <= ymax; ++j) {
      const std::vector<int> &pindices = grid(i, j).particles;
      for (int pind : pindices) {
        if (pind != -1)
          f(particles[pind]);
      }
    }
}

} // poseidon namespace
