#pragma once

#include <ponos.h>

namespace poseidon {

  class Particle2D {
  public:
    Particle2D(ponos::Point2 _p, ponos::Vector2 _v) {
      p = _p;
      v = _v;
    }
    ponos::Point2 p;
    ponos::Vector2 v;
  };

  class Particle {
  public:
    Particle(ponos::Point3 _p, ponos::Vector3 _v) {
      p = _p;
      v = _v;
    }
    ponos::Point3 p;
    ponos::Vector3 v;
  };

} // poseidon namespace
