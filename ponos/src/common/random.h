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
			virtual float randomFloat() { return 0.f; }
			/* pseudo-rangom integer number.
			 *
			 * @return int in the range [0, 2^32)
			 */
			virtual ulong randomInt() { return 0; }
			/* pseudo-rangom integer number.
			 *
			 * @return unsigned int in the range [0, 2^32)
			 */
			virtual ulong randomUInt() { return 0; }

			virtual ~RNG() {}
	};

	/* Random Number Generator
	 * Implements the "Halton Sequence".
	 */
	class HaltonSequence : public RNG {
		public:
			HaltonSequence(uint b)
				: base(b), ind(1) {}
			/* @inherit */
			float randomFloat() override {
				float result = 0.f;
				float f = 1.f;
				uint i = ind++;
				while(i > 0) {
					f /= base;
					result += f * (i % base);
					i /= base;
				}
				return result;
			}

		private:
			uint base, ind;
	};

} // ponos namespace

#endif // PONOS_COMMON_RANDOM_H

