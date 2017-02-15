#ifndef HELIOS_FILM_IMAGE_H
#define HELIOS_FILM_IMAGE_H

#include "core/film.h"
#include "core/filter.h"

#include <ponos.h>

#include <string>

namespace helios {

	/* Film implementation
	 * Implementation of <helios::Film> that filters image sample values with a given reconstruction filter.
	 */
	class ImageFilm : public Film {
		public:
			/* Constructor
			 * @xres **[in]** image resolution in x (in pixels)
			 * @yres **[in]** image resolution in y (in pixels)
			 * @f **[out]** filter function
			 * @crop[4] **[in]** subrectangle of the pixels to be rendered (NDC space **[0, 1]**)
			 * @fn **[in]** output image filename
			 */
			ImageFilm(int xres, int yres, Filter* f, const float crop[4], const std::string& fn);
			virtual ~ImageFilm() {}
			/* @inherit */
			void addSample(const CameraSample &sample, const Spectrum &L);

			Filter* filter;
			float cropWindow[4];
			std::string filename;

		private:
			int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
			struct Pixel {
				Pixel() {
					for(int i = 0; i < 3; i++)
						Lxyz[i] = splatXYZ[i] = 0.f;
					weightSum = 0.f;
				}
				float weightSum;
				// radiance
				float Lxyz[3];
				// unweighted sample contributions
				float splatXYZ[3];
				// ensure this is struct is 32bit large
				float pad;
			};
			ponos::BlockedArray<Pixel> *pixels;
			float *filterTable;
	};

} // helios namespace

#endif // HELIOS_FILM_IMAGE_H

