#include "structures/particle_grid.h"

namespace poseidon {
    ParticleGrid::ParticleGrid() {}

    void ParticleGrid::set(uint32_t w, uint32_t h, Vector2 offset, Vector2 cellSize) {
        grid.setDimensions(w, h);
        grid.setTransform(offset, cellSize);
    }
} // poseidon namespace
