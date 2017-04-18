#pragma once

#include <ponos.h>

namespace poseidon {

enum class ParticleTypes { FLUID, AIR, SOLID, OTHER };

enum class ParticleAttribute {
  POSITION,
  POSITION_X,
  POSITION_Y,
  POSITION_Z,
  VELOCITY,
  VELOCITY_X,
  VELOCITY_Y,
  VELOCITY_Z,
  DENSITY
};

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
 * Represents a little blob of fluid carring quantities such as velocity and
 * mass.
 */
struct Particle {
  ponos::Point3 position;
  ponos::vec3 velocity;
  ponos::vec3 normal;
  float density;
  float mass;
  bool invalid;
  ParticleTypes type;
};

struct FLIPParticle2D {
  FLIPParticle2D() = default;
  FLIPParticle2D(const ponos::Point2 &p) : position(p) {
    mass = 1.f;
    type = ParticleTypes::FLUID;
  }
  ~FLIPParticle2D() {}
  ponos::Point2 position;
  ponos::vec2 velocity;
  float density;
  float mass;
  float radius;
  bool invalid;
  ParticleTypes type;
  ponos::Normal normal;
  int meshId;
  ponos::Point2 p_copy;
  ponos::vec2 v_copy;
};

} // poseidon namespace
