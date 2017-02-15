#include "core/spectrum.h"

namespace helios {

	SampledSpectrum SampledSpectrum::fromRGB(const float rgb[3],
			SpectrumType type) {
		SampledSpectrum r;
		if (type == SpectrumType::REFLECTANCE) {
			// Convert reflectance spectrum to RGB
			if (rgb[0] <= rgb[1] && rgb[0] <= rgb[2]) {
				// Compute reflectance _SampledSpectrum_ with _rgb[0]_ as minimum
				r += rgb[0] * rgbRefl2SpectWhite;
				if (rgb[1] <= rgb[2]) {
					r += (rgb[1] - rgb[0]) * rgbRefl2SpectCyan;
					r += (rgb[2] - rgb[1]) * rgbRefl2SpectBlue;
				} else {
					r += (rgb[2] - rgb[0]) * rgbRefl2SpectCyan;
					r += (rgb[1] - rgb[2]) * rgbRefl2SpectGreen;
				}
			} else if (rgb[1] <= rgb[0] && rgb[1] <= rgb[2]) {
				// Compute reflectance _SampledSpectrum_ with _rgb[1]_ as minimum
				r += rgb[1] * rgbRefl2SpectWhite;
				if (rgb[0] <= rgb[2]) {
					r += (rgb[0] - rgb[1]) * rgbRefl2SpectMagenta;
					r += (rgb[2] - rgb[0]) * rgbRefl2SpectBlue;
				} else {
					r += (rgb[2] - rgb[1]) * rgbRefl2SpectMagenta;
					r += (rgb[0] - rgb[2]) * rgbRefl2SpectRed;
				}
			} else {
				// Compute reflectance _SampledSpectrum_ with _rgb[2]_ as minimum
				r += rgb[2] * rgbRefl2SpectWhite;
				if (rgb[0] <= rgb[1]) {
					r += (rgb[0] - rgb[2]) * rgbRefl2SpectYellow;
					r += (rgb[1] - rgb[0]) * rgbRefl2SpectGreen;
				} else {
					r += (rgb[1] - rgb[2]) * rgbRefl2SpectYellow;
					r += (rgb[0] - rgb[1]) * rgbRefl2SpectRed;
				}
			}
			r *= .94;
		} else {
			// Convert illuminant spectrum to RGB
			if (rgb[0] <= rgb[1] && rgb[0] <= rgb[2]) {
				// Compute illuminant _SampledSpectrum_ with _rgb[0]_ as minimum
				r += rgb[0] * rgbIllum2SpectWhite;
				if (rgb[1] <= rgb[2]) {
					r += (rgb[1] - rgb[0]) * rgbIllum2SpectCyan;
					r += (rgb[2] - rgb[1]) * rgbIllum2SpectBlue;
				} else {
					r += (rgb[2] - rgb[0]) * rgbIllum2SpectCyan;
					r += (rgb[1] - rgb[2]) * rgbIllum2SpectGreen;
				}
			} else if (rgb[1] <= rgb[0] && rgb[1] <= rgb[2]) {
				// Compute illuminant _SampledSpectrum_ with _rgb[1]_ as minimum
				r += rgb[1] * rgbIllum2SpectWhite;
				if (rgb[0] <= rgb[2]) {
					r += (rgb[0] - rgb[1]) * rgbIllum2SpectMagenta;
					r += (rgb[2] - rgb[0]) * rgbIllum2SpectBlue;
				} else {
					r += (rgb[2] - rgb[1]) * rgbIllum2SpectMagenta;
					r += (rgb[0] - rgb[2]) * rgbIllum2SpectRed;
				}
			} else {
				// Compute illuminant _SampledSpectrum_ with _rgb[2]_ as minimum
				r += rgb[2] * rgbIllum2SpectWhite;
				if (rgb[0] <= rgb[1]) {
					r += (rgb[0] - rgb[2]) * rgbIllum2SpectYellow;
					r += (rgb[1] - rgb[0]) * rgbIllum2SpectGreen;
				} else {
					r += (rgb[1] - rgb[2]) * rgbIllum2SpectYellow;
					r += (rgb[0] - rgb[1]) * rgbIllum2SpectRed;
				}
			}
			r *= .86445f;
		}
		return r.clamp();
	}

} // helios namespace
