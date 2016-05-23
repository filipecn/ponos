#pragma once

#include "elements/particle.h"

#include <vector>

namespace poseidon {

    class ParticleSet {
    protected:
        std::vector<Particle> particles;
    };

} // poseidon namespace
