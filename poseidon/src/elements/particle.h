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

	/* element
	 * Represents a little blob of fluid carring quantities such as velocity and mass.
	 */
	struct Particle {
		ponos::Point3 position;
		ponos::Vector3 velocity;
		float density;
		float mass;
	};

} // poseidon namespace
