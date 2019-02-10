#ifndef HELIOS_CORE_SAMPLER_H
#define HELIOS_CORE_SAMPLER_H

#include "core/intersection.h"
#include "core/spectrum.h"
#include "geometry/h_ray.h"

#include <ponos.h>

#include <vector>

namespace helios {

	//////////////////TEMP//////////////////////////
	class VolumeIntegrator {};
	class SurfaceIntegrator {};
	class Scene {};
	//////////////////TEMP//////////////////////////
	class Sample;

	/* Sampler interface.
	 * Generates a set of multidimensional sample positions per pixel.
	 * The samples are generated from PixelStart to PixelEnd - 1, inclusive, and the time values will be in range of shutterOpen and shutterClose.
	 */
	class Sampler {
		public:
		/* Constructor.
		 * @xstart first pixel coordinate on X axis
		 * @xend (last + 1) pixel coordinate on X axis
		 * @ystart first pixel coordinate on Y axis
		 * @yend (last + 1) pixel coordinate on Y axis
		 * @spp number of samples per pixel
		 * @sopen camera's shutter open time
		 * @sclose camera's shutter close time
		 */
		Sampler(int xstart, int xend, int ystart, int yend, int spp, float sopen, float sclose);
		/* generate more samples.
		 * @samples **[out]** this function will fill samples with the new samples
		 * @rng **[in]** the pseudo-random number generator
		 *
		 * @return the number of generated samples.
		 */
		virtual int getMoreSamples(Sample *samples, ponos::RNG &rng) = 0;
		/* max number of samples it can generates in one call.
		 * help the caller of <getMoreSamples> to allocate memory for the sample.
		 */
		virtual int maximumSampleCount() = 0;
		/* report
     * @samples collection of samples (originally from getModeSamples())
		 * @rays [out] which rays were generated
		 * @Ls [out] which radiance values were computed
		 * @isects [out] which intersections were found
		 * @count number of samples
		 *
		 * This method is usefull for adaptive sampling algorithms.
		 *
		 * @return true if the sample values should be added to the image being generated
		 */
		bool reportResults(Sample *samples, const RayDifferential *rays, const Spectrum *Ls, const Intersection *isects, int count);
		/* creates a new Sampler.
		 * @num ranges from 0 to count-1
		 * @count total number of subsamplers being used
		 *
		 * Creates a new Sampler responsible for a subset of the image.
		 */
		virtual Sampler* getSubSampler(int num, int count) = 0;
		/* Generate samples for a tile.
		 * @num tile number
		 * @count total number of tiles
		 * @newXStart first pixel coordinate on X axis
		 * @newXEnd (last + 1) pixel coordinate on X axis
		 * @newYStart first pixel coordinate on Y axis
		 * @newYEnd (last + 1) pixel coordinate on Y axis
		 *
		 * A lot of implementations decompose the image into rectangular tiles and have each subsampler generate samples for each tile. computeSubWindow() computes a pixel sampling range for a specific tile.
		 */
		void computeSubWindow(int num, int count, int *newXStart, int *newXEnd, int *newYStart, int *newYEnd) const;
		/* Computes the number of samples after rounding to size. // TODO
		 * @size size to be rounded to
		 *
		 * @return the number of samples that should be requested for this size
		 */
		virtual int roundSize(int size) const = 0;

		const int xPixelStart, xPixelEnd, yPixelStart, yPixelEnd;
		const int samplesPerPixel;
		const float shutterOpen, shutterClose;
	};



	/* Sample implementation.
	 *
	 * The sample get from the integrators the requests for samples. The integrators can ask for multiple 1D and 2D sampling patterns.
	 */
	struct Sample : public CameraSample {
		/* Constructor.
		 * @sampler
		 * @surf Surface integrator
		 * @vol Volume integrator
		 * @scene
		 *
		 * Calls <Integrator::requestSamples> methods of the surface and volume integrators to find out what samples they will need.
		 */
		Sample(Sampler *sampler, SurfaceIntegrator *surf, VolumeIntegrator *vol, const Scene *scene);
		/* Request 1D sample pattern.
		 * @num number of samples
		 *
		 * @return array index that can be used to access this sample value later
		 */
		uint32 add1D(uint32 num) {
			n1D.emplace_back(num);
			return n1D.size() - 1;
		}
		/* Request 2D sample pattern.
		 * @num number of samples
		 *
		 * @return array index that can be used to access this sample value later
		 */
		uint32 add2D(uint32 num) {
			n2D.emplace_back(num);
			return n2D.size() - 1;
		}
		/* allocates sample memory.
		 */
		void allocateSampleMemory();
		/* generate clone instances
		 * @count number of copies
		 *
		 * @return array of copies
		 */
		Sample* duplicate(int count) const;

		~Sample() { // TODO
		}

		std::vector<uint32> n1D, n2D;

		// the integrator can retrieve the sample value by oneD[<sampleOffset>][i]
		float **oneD, **twoD;
	};

	/* generate
	 * @samp **[out]** samples data
	 * @nSamples **[in]** number of samples
	 * @rng **[out]** the pseudo-random number generator
	 * @jitter **[in]** if true, the samples are jittered
	 *
	 * Generates a stratified 1D samples pattern
	 */
	void generateStratifiedSample1D(float *samp, int nSamples, ponos::RNG& rng, bool jitter);
	/* generate
	 * @samp **[out]** samples data
	 * @nx **[in]** number of samples in X axis
	 * @ny **[in]** number of samples in Y axis
	 * @rng **[out]** the pseudo-random number generator
	 * @jitter **[in]** if true, the samples are jittered
	 *
	 * Generates a stratified 2D samples pattern
	 */
	void generateStratifiedSample2D(float *samp, int nx, int ny, ponos::RNG& rng, bool jitter);
	/* generate
	 * @samples **[out]** samples data
	 * @nSamples **[in]** number of samples
	 * @nDim **[in]** number of dimensions
	 * @rng **[out]** the pseudo-random number generator
	 *
	 * Generates **samples** in an arbitrary dimension **nDim** using LHS (Latin Hypercube Sampling). LHS divides each dimension's axis into _n_ regions and generates a jittered sample in each region along the diagonal, then shuffles the samples in each dimension.
	 */
	void generateLatinHypercube(float *samples, uint32 nSamples, uint32 nDim, ponos::RNG& rng);
} // helios namespace

#endif // HELIOS_CORE_SAMPLER_H
