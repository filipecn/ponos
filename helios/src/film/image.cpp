#include "film/image.h"

#include <ponos.h>

namespace helios {

	ImageFilm::ImageFilm(int xres, int yres, Filter* f, const float crop[4], const std::string& fn)
	: Film(xres, yres) {
		filter = f;
		memcpy(cropWindow, crop, 4 * sizeof(float));
		filename = fn;
		// Compute film image extent
		xPixelStart = ponos::ceil2Int(xResolution * cropWindow[0]);
		xPixelCount = std::max(1, ponos::ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
		yPixelStart = ponos::ceil2Int(yResolution * cropWindow[2]);
		yPixelCount = std::max(1, ponos::ceil2Int(yResolution * cropWindow[3]) - yPixelStart);
		// Allocate film image storage
		pixels = new ponos::BlockedArray<Pixel>(xPixelCount, yPixelCount);
		// Precompute filter weight table
#define FILTER_TABLE_SIZE 16
		filterTable = new float[SQR(FILTER_TABLE_SIZE)];
		float *ftp = filterTable;
		for(int y = 0; y < FILTER_TABLE_SIZE; ++y) {
			float fy = (static_cast<float>(y) + .5f) * filter->yWidth / FILTER_TABLE_SIZE;
			for(int x = 0; x < FILTER_TABLE_SIZE; ++x) {
				float fx = (static_cast<float>(x) + .5f) * filter->xWidth / FILTER_TABLE_SIZE;
				*ftp++ = filter->evaluate(fx, fy);
			}
		}
	}

	void ImageFilm::addSample(const CameraSample &sample, const Spectrum &L) {
		// Compute sample's raster extent
		float dimageX = sample.imageX - 0.5f;
		float dimageY = sample.imageY - 0.5f;
		int x0 = ponos::ceil2Int(dimageX - filter->xWidth);
		int x1 = ponos::floor2Int(dimageX + filter->xWidth);
		int y0 = ponos::ceil2Int(dimageY - filter->yWidth);
		int y1 = ponos::floor2Int(dimageY + filter->yWidth);
		x0 = std::max(x0, xPixelStart);
		x1 = std::min(x1, xPixelStart + xPixelCount - 1);
		y0 = std::max(y0, yPixelStart);
		y1 = std::min(y1, xPixelStart + yPixelCount - 1);
		if((x1 - x0) < 0 || (y1 - y0) < 0)
			return;
		// Loop over filter support and add sample to pixel arrays
		float xyz[3];
		L.toXYZ(xyz);
		// precompute x and y filter table offsets
		int *ifx = ALLOCA(int, x1 - x0 + 1);
		for(int x = x0; x <= x1; ++x) {
			float fx = fabsf((x - dimageX) * filter->invXWidth * FILTER_TABLE_SIZE);
			ifx[x - x0] = std::min(ponos::floor2Int(fx), FILTER_TABLE_SIZE - 1);
		}
		int *ify = ALLOCA(int, y1 - y0 + 1);
		for(int y = y0; y <= y1; ++y) {
			float fy = fabsf((y - dimageY) * filter->invYWidth * FILTER_TABLE_SIZE);
			ify[y - y0] = std::min(ponos::floor2Int(fy), FILTER_TABLE_SIZE - 1);
		}
		bool syncNeeded = (filter->xWidth > 0.5f || filter->yWidth > 0.5f);
		for(int y = y0; y <= y1; ++y) {
			for(int x = x0; x <= x1; ++x) {
				// Evaluate filter at (x, y) pixel
				int offset = ify[y - y0] * FILTER_TABLE_SIZE + ifx[x - x0];
				float filterWeight = filterTable[offset];
				// Update pixel values filtered sample contribution
				Pixel &pixel = (*pixels)(x - xPixelStart, y - yPixelStart);
				if(!syncNeeded) {
					pixel.Lxyz[0] += filterWeight * xyz[0];
					pixel.Lxyz[1] += filterWeight * xyz[1];
					pixel.Lxyz[2] += filterWeight * xyz[2];
					pixel.weightSum += filterWeight;
				} else {
					// Safely update Lxyz and weightSum even with concurrency
					// TODO pg 411
				}
			}
		}
	}
} // helios namespace
