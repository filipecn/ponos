#pragma once

#include "elements/particle.h"

#include <ponos.h>
#include <vector>

namespace poseidon {

  template<class T>
  class ParticleSet {
  public:
    const std::vector<T>& getParticles() { return particles; }
    T& getParticleReference(uint32_t i) {
      return particles[i];
    }
    T getParticle(uint32_t i) const {
      return particles[i];
    }
    virtual void setPos(uint32_t i, ponos::Point2 p) {
      particles[i].p = p;
    }
    virtual uint32_t addParticle(const T& p) {
      particles.emplace_back(p);
      return particles.size();
    }
    int size() {
      return particles.size();
    }
  protected:
    std::vector<T> particles;
  };

} // poseidon namespace
