#ifndef PONOS_COMMON_RANDOM_H
#define PONOS_COMMON_RANDOM_H

#include <ponos/common/defs.h>
#include <ponos/geometry/numeric.h>
#include <ponos/geometry/bbox.h>

namespace ponos {

/** \brief Random Number Generator
 * Implements the "Mersenne Twister" by Makoto Matsumoto and Takuji Nishimura.
 */
class RNG {
public:
  /// \param seed
  explicit RNG(uint32 seed = 0) { UNUSED_VARIABLE(seed); }
  /// \param seed
  void setSeed(uint32 seed) { UNUSED_VARIABLE(seed); }
  /// pseudo-random floating-point number.
  /// \return a float in the range [0, 1)
  virtual float randomFloat() { return 0.f; }
  /// pseudo-random integer number.
  /// \return int in the range [0, 2^32)
  virtual ulong randomInt() { return 0; }
  /// \brief pseudo-rangom integer number.
  /// \return unsigned int in the range [0, 2^32)
  virtual ulong randomUInt() { return 0; }
  virtual ~RNG() = default;
};

/** \brief Random Number Generator
 * Implements the "Halton Sequence".
 */
class HaltonSequence : public RNG {
public:
  /// Default constructor.
  HaltonSequence() : base(2), ind(1) {}
  /// \param b base ( > 1)
  explicit HaltonSequence(uint b) : base(b), ind(1) {}
  /// \param b base ( > 1)
  void setBase(uint b) {
    base = b;
    ind = 1;
  }
  /// pseudo-random floating-point number.
  /// \return a float in the range [0, 1)
  float randomFloat() override {
    float result = 0.f;
    float f = 1.f;
    uint i = ind++;
    while (i > 0) {
      f /= base;
      result += f * (i % base);
      i /= base;
    }
    return result;
  }
  /// \param a lower bound
  /// \param b upper bound
  /// \return random float in the range [a,b)
  float randomFloat(float a, float b) { return lerp(randomFloat(), a, b); }

private:
  uint base, ind;
};

class RNGSampler {
public:
  /// \param rng random number generator
  explicit RNGSampler(RNG *rngX, RNG *rngY, RNG *rngZ) :
      rngX_(rngX), rngY_(rngY), rngZ_(rngZ) {}
  /// Samples a bbox region
  /// \param region sampling domain
  /// \return a random point inside **region**
  Point3 sample(const BBox &region) {
    return Point3(rngX_->randomFloat() * region.size(0) + region.pMin[0],
                  rngY_->randomFloat() * region.size(1) + region.pMin[1],
                  rngZ_->randomFloat() * region.size(2) + region.pMin[2]);
  }
private:
  std::shared_ptr<RNG> rngX_;
  std::shared_ptr<RNG> rngY_;
  std::shared_ptr<RNG> rngZ_;
};

} // ponos namespace

#endif // PONOS_COMMON_RANDOM_H
