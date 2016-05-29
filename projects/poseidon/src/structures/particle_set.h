#pragma once

#include "elements/particle.h"

#include <vector>

namespace poseidon {

  template<class T>
  class ParticleSet {
  public:
    const std::vector<T>& getParticles() { return particles; }
    virtual uint32_t addParticle(const T& p) {
      particles.emplace_back(p);
      return particles.size();
    }
  protected:
    std::vector<T> particles;
  };

} // poseidon namespace
