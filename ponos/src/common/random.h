#ifndef PONOS_COMMON_RANDOM_H
#define PONOS_COMMON_RANDOM_H

#include "common/defs.h"

namespace ponos {

	/* Random Number Generator
	 * Implements the "Mersenne Twister" by Makoto Matsumoto and Takuji Nishimura.
	 */
	class RNG {
  	public:
	 		RNG(uint32 seed = 0) {}
			void setSeed(uint32 seed) {}
			/* pseudo-random floating-point number.
			 *
			 * @return a float in the range [0, 1)
			 */
			float randomFloat() const { return 0.f; }
			/* pseudo-rangom integer number.
			 *
			 * @return int in the range [0, 2^32)
			 */
			ulong randomInt() const { return 0; }
			virtual ~RNG() {}
	};

} // ponos namespace

#endif // PONOS_COMMON_RANDOM_H

