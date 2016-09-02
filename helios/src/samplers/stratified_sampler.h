#ifndef HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H
#define HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H

#include "core/sampler.h"

namespace helios {

	/* sampler
	 * Divides the image into rectangular regions and generates  a single sample inside each region (_stratum_).
	 */
	class StratifiedSampler : public Sampler {
		public:
			/* Constructor.
			 * @xstart **[in]** first pixel coordinate on X axis
			 * @xend **[in]** (last + 1) pixel coordinate on X axis
			 * @ystart **[in]** first pixel coordinate on Y axis
			 * @yend **[in]**(last + 1) pixel coordinate on Y axis
			 * @xs **[in]** number of _strata_ in X axis
			 * @ys **[in]** number of _strata_ in Y axis
			 * @jitter **[in]** if true, the samples are jittered
			 * @sopen **[in]** camera's shutter open time
			 * @sclose **[in]** camera's shutter close time
			 */
			StratifiedSampler(int xstart, int xend, int ystart, int yend, int xs, int ys, bool jitter, float sopen, float sclose);
			virtual ~StratifiedSampler() {}
			/* @inherit */
			int getMoreSamples(Sample *samples, ponos::RNG &rng) override;
			/* @inherit */
			int maximumSampleCount() override { return xPixelSamples * yPixelSamples; }
			/* @inherit
			 * @return null if input is degenerated
			 **/
			Sampler* getSubSampler(int num, int count) override;
			/* @inherit */
			int roundSize(int size) const override { return size; }
		private:
			int xPixelSamples, yPixelSamples;
			bool jitterSamples;
			int xPos, yPos;
			float *buffer;
	};

} // helios namespace

#endif // HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H

