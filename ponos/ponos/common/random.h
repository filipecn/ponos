/** @ingroup Common */
#ifndef PONOS_COMMON_RANDOM_H
#define PONOS_COMMON_RANDOM_H

#include <ponos/common/defs.h>
#include <ponos/geometry/numeric.h>

namespace ponos {

/** \brief Random Number Generator
 * Implements the "Mersenne Twister" by Makoto Matsumoto and Takuji Nishimura.
 */
class RNG {
public:
  /** \brief  Default constructor.
   * \param seed
   */
  RNG(uint32 seed = 0) { UNUSED_VARIABLE(seed); }
  /** \brief
   * \param seed
   */
  void setSeed(uint32 seed) { UNUSED_VARIABLE(seed); }
  /** \brief pseudo-random floating-point number.
   * \return a float in the range [0, 1)
   */
  virtual float randomFloat() { return 0.f; }
  /** \brief pseudo-rangom integer number.
   *
   * \return int in the range [0, 2^32)
   */
  virtual ulong randomInt() { return 0; }
  /** \brief pseudo-rangom integer number.
   *
   * \return unsigned int in the range [0, 2^32)
   */
  virtual ulong randomUInt() { return 0; }

  virtual ~RNG() {}
};

/** \brief Random Number Generator
 * Implements the "Halton Sequence".
 */
class HaltonSequence : public RNG {
public:
  /** \brief Default constructor.
   */
  HaltonSequence() : base(2), ind(1) {}
  /** \brief Constructor.
         * \param b base ( > 1)
         */
  HaltonSequence(uint b) : base(b), ind(1) {}
  /** \brief
   * \param b base ( > 1)
   */
  void setBase(uint b) {
    base = b;
    ind = 1;
  }
  /* @inherit */
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

  float randomFloat(float a, float b) { return lerp(randomFloat(), a, b); }

private:
  uint base, ind;
};

} // ponos namespace

#endif // PONOS_COMMON_RANDOM_H
       /**@}*/
