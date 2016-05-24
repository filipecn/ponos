#include "algorithms/flip.h"

namespace poseidon {

  void FLIP::set(uint32_t w, uint32_t h, vec2 offset, float scale) {
    dx = scale;
    particleGrid.set(w, h, offset, vec2(scale, scale));
  }

} // poseidon namespace
