#ifndef HELIOS_CORE_FILM_H
#define HELIOS_CORE_FILM_H

#include "core/sampler.h"

namespace helios {

	/* interface
	 * DIctates how the incident light is acctually transformed into colors in an image.
	 */
	class Film {
		public:
			/* Constructor
			 * @xres **[in]** image resolution in x (in pixels)
			 * @yres **[in]** image resolution in y (in pixels)
			 */
			Film(int xres, int yres)
				: xResolution(xres), yResolution(yres) {}
			virtual ~Film() {}

			/* add sample
			 * @sample **[in]** camera sample
			 * @L **[in]** spectrum
			 *
			 * Adds a sample to contribute a pixel's final value. Usually a pixel value is a weighted average of the nearby samples.
			 *
			 */
			virtual void addSample(const CameraSample& sample, const Spectrum& L) = 0;
			/* splat
			 * @sample **[in]** camera sample
			 * @L **[in]** spectrum
			 *
			 * Different from <addSample> method, splat computes the final value of a pixel as a summed contribution of the nearby samples.
			 */
			virtual void splat(const CameraSample& sample, const Spectrum& L) = 0;
			/* get extents
			 * @xstart **[out]** first pixel in X
			 * @xend **[out]** last + 1 pixel in X
			 * @ystart **[out]**ix first pixel in Y
			 * @yend **[out]** last + 1 pixel in Y
			 * In some cases it is necessary to sample the image outside of its resolution limits, this method returns the new pixel ranges.
			 */
			virtual void getSampleExtent(int* xstart, int* xend, int* ystart, int* yend) const = 0;
			/* get extents
			 * @xstart **[out]** first pixel in X
			 * @xend **[out]** last + 1 pixel in X
			 * @ystart **[out]**ix first pixel in Y
			 * @yend **[out]** last + 1 pixel in Y
			 * Provides the range of pixels in the actual image, in other words, without <getSampleExtent>.
			 */
			virtual void getPixelExtent(int* xstart, int* xend, int* ystart, int* yend) const = 0;
			/* notification
			 * @x0 **[in]** fisrt pixel in X of the updated region
			 * @y0 **[in]** first pixel in Y of the updated region
			 * @x1 **[in]** last pixel in X of the updated region
			 * @y1 **[in]** last pixel in Y of the updated region
			 * @splatScale **[in]** splat scale
			 * Some implementations of **Film** need to be notified that a region of pixels has recently been updated.
			 */
			void updateDisplay(int x0, int y0, int x1, int y1, float splatScale) {}
			/* finalize image
			 * @splatScale **[in]** scale factor for the samples provided to the **splat** method
			 * Process the final image and stores it in a file.
			 */
			virtual void writeImage(float splatScale = 1.f) = 0;

			const int xResolution, yResolution;
	};

} // helios namespace

#endif // HELIOS_CORE_FILM_H

