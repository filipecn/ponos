#include "algorithms/flip.h"

namespace poseidon {

  void FLIP::set(uint32_t w, uint32_t h, vec2 offset, float scale) {
    dx = scale;
    vec2 vscale = vec2(scale, scale);
    particleGrid.set(w, h, offset, vscale);
    u.set(w + 1, h, offset + vec2(-0.5f * dx, 0.f), vscale);
    u.init();
    v.set(w, h + 1, offset + vec2(0.f, -0.5f * dx), vscale);
    u.init();
    p.set(w, h, offset, vscale);
    p.init();
  }

  void FLIP::fillCell(uint32_t i, uint32_t j) {
    particleGrid.fillCell(i, j);
  }

  void FLIP::step() {

  }

  void FLIP::gather(ZGrid<float> &grid) {
    for (int i = 0; i < grid.width; ++i)
    for (int j = 0; j < grid.height; ++j) {
      Point2 wp = grid.toWorld(Point2(i, j));
      Point2 wpmin = wp + vec2(-dx, -dx);
      Point2 wpmax = wp + vec2(dx, dx);
      particleGrid.iterateParticles(ponos::BBox2D(wpmin, wpmax),
      [](const Particle2D &){});
    }
  }

} // poseidon namespace
