#ifndef HELIOS_CORE_SPECTRUM_H
#define HELIOS_CORE_SPECTRUM_H

#include <ponos.h>

#include <algorithm>
#include <cmath>
#include <map>

namespace helios {

	static const int nCIESamples = 471;
	extern const float CIE_X[nCIESamples];
	extern const float CIE_Y[nCIESamples];
	extern const float CIE_Z[nCIESamples];
	extern const float CIE_lambda[nCIESamples];

	static const int nRGB2SpectSamples = 32;
	extern const float RGB2SpectLambda[nRGB2SpectSamples];
	extern const float RGBRefl2SpectWhite[nRGB2SpectSamples];
	extern const float RGBRefl2SpectCyan[nRGB2SpectSamples];
	extern const float RGBRefl2SpectMagenta[nRGB2SpectSamples];
	extern const float RGBRefl2SpectYellow[nRGB2SpectSamples];
	extern const float RGBRefl2SpectRed[nRGB2SpectSamples];
	extern const float RGBRefl2SpectGreen[nRGB2SpectSamples];
	extern const float RGBRefl2SpectBlue[nRGB2SpectSamples];

	extern const float RGBIllum2SpectWhite[nRGB2SpectSamples];
	extern const float RGBIllum2SpectCyan[nRGB2SpectSamples];
	extern const float RGBIllum2SpectMagenta[nRGB2SpectSamples];
	extern const float RGBIllum2SpectYellow[nRGB2SpectSamples];
	extern const float RGBIllum2SpectRed[nRGB2SpectSamples];
	extern const float RGBIllum2SpectGreen[nRGB2SpectSamples];
	extern const float RGBIllum2SpectBlue[nRGB2SpectSamples];

	/* convert
	 * @xyz **[in]** xyz values
	 * @rgb **[out]** returned rgb values
	 */
	inline void XYZToRGB(const float xyz[3], float rgb[3]) {
		rgb[0] =  3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
		rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
		rgb[2] =  0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
	}

	/* convert
	 * @rgb **[in]** rgb values
	 * @xyz **[out]** returned xyz values
	 */
	inline void RGBToXYZ(const float rgb[3], float xyz[3]) {
		xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
		xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
		xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
	}

	/* base class
	 * Represents a spectrum by a particular number of samples (given by **nSamples** parameter) of the SPD(_spectral power distribution_).
	 */
	template <int nSamples>
		class CoefficientSpectrum {
			public:
				/* Constructor.
				 * @v **[in | optional]** constant value across all wavelenghts
				 */
				CoefficientSpectrum(float v = 0.f) {
					for(int i = 0; i < nSamples; i++)
						c[i] = v;
				}
				virtual ~CoefficientSpectrum() {}

				CoefficientSpectrum& operator+=(const float &f) {
					for(int i = 0; i < nSamples; i++)
						c[i] += f;
					return *this;
				}
				CoefficientSpectrum operator+(const float &f) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] += f;
					return r;
				}
				CoefficientSpectrum& operator+=(const CoefficientSpectrum &s2) {
					for(int i = 0; i < nSamples; i++)
						c[i] += s2.c[i];
					return *this;
				}
				CoefficientSpectrum operator+(const CoefficientSpectrum &s2) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] += s2.c[i];
					return r;
				}
				CoefficientSpectrum& operator*=(const CoefficientSpectrum &s2) {
					for(int i = 0; i < nSamples; i++)
						c[i] *= s2.c[i];
					return *this;
				}
				CoefficientSpectrum operator*(const CoefficientSpectrum &s2) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] *= s2.c[i];
					return r;
				}
				friend inline CoefficientSpectrum operator*(const float &f, const CoefficientSpectrum &s) {
					return s * f;
				}
				CoefficientSpectrum& operator*=(const float &f) {
					for(int i = 0; i < nSamples; i++)
						c[i] *= f;
					return *this;
				}
				CoefficientSpectrum operator*(const float &f) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] *= f;
					return r;
				}
				CoefficientSpectrum& operator/=(const CoefficientSpectrum &s2) {
					for(int i = 0; i < nSamples; i++)
						c[i] /= s2.c[i];
					return *this;
				}
				CoefficientSpectrum operator/(const CoefficientSpectrum &s2) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] /= s2.c[i];
					return r;
				}
				CoefficientSpectrum& operator/=(const float &f) {
					for(int i = 0; i < nSamples; i++)
						c[i] /= f;
					return *this;
				}
				CoefficientSpectrum operator/(const float &f) const {
					CoefficientSpectrum r = *this;
					for(int i = 0; i < nSamples; i++)
						r.c[i] /= f;
					return r;
				}
				/* query
				 * Useful to ray tracing algorithms to avoid the computation of reflected rays of zero reflectance surfaces.
				 */
				bool isBlack() const {
					for(int i = 0; i < nSamples; i++)
						if(c[i] != 0.) return false;
					return true;
				}
				/* clamp
				 * @l **[in | optional]** low (default = 0)
				 * @h **[in | optional]** high (default = INFINITY)
				 * @return the spectrum with values clamped to be in **[l, h]**
				 */
				CoefficientSpectrum clamp(float l = 0, float h = INFINITY) const {
					CoefficientSpectrum r;
					for(int i = 0; i < nSamples; i++)
						r.c[i] = ponos::clamp(c[i], l, h);
					return r;
				}
				bool hasNaNs() const {
					for(int i = 0; i < nSamples; i++)
						if(std::isnan(c[i]))
							return true;
					return false;
				}

				friend CoefficientSpectrum sqrt(const CoefficientSpectrum &s) {
					CoefficientSpectrum r;
					for(int i = 0; i < nSamples; i++)
						r.c[i] = sqrtf(s.c[i]);
					return r;
				}

			protected:
				float c[nSamples];
		};

	static const int sampledLambdaStart = 400;
	static const int sampledLambdaEnd = 700;
	static const int nSpectralSamples = 30;

	enum class SpectrumType { REFLECTANCE, ILLUMINANT };

	/* average
	 * @lambda **[in]** coordinates
	 * @vals **[in]** values on coordinates
	 * @n **[in]** number of samples
	 * @lambdaStart **[in]** first coordinate
	 * @lambdaEnd **[in]** last coordinate
	 * Computes the average value of a list of samples.
	 * @return average
	 */
	float averageSpectrumSamples(const float *lambda, const float *vals, int n, float lambdaStart, float lambdaEnd) {
		if(lambdaEnd <= lambda[0]) return vals[0];
		if(lambdaStart >= lambda[n - 1]) return vals[n - 1];
		if(n == 1) return vals[0];
		float sum = 0.f;
		// Add contributions of constant sefments
		if(lambdaStart < lambda[0])
			sum += vals[0] * (lambda[0] - lambdaStart);
		if(lambdaEnd > lambda[n - 1])
			sum += vals[n - 1] * (lambdaEnd - lambda[n - 1]);
		// Advance to first relevant wavelenght segment
		int i = 0;
		while(lambdaStart > lambda[n + 1]) i++;
		// Loop over wavelenght sample segments
#define INTERP(w, i) \
		ponos::lerp(((w) - lambda[i]) / (lambda[(i) + 1] - lambda[i]), vals[i], vals[(i) + 1])
#define SEG_AVG(wl0, wl1, i) (0.5f * (INTERP(wl0, i) + INTERP(wl1, i)))
		for(; i + 1 < n && lambdaEnd >= lambda[i]; i++) {
			float segStart = std::max(lambdaStart, lambda[i]);
			float segEnd = std::min(lambdaEnd, lambda[i + 1]);
			sum += SEG_AVG(segStart, segEnd, i) * (segEnd - segStart);
		}
#undef INTERP
#undef SEG_AVG
		return sum / (lambdaEnd - lambdaStart);
	}
	/* interpolation
	 * @lambda **[in]** coordinates
	 * @vals **[in]** values on coordinates
	 * @n **[in]** number of samples
	 * @lambdaStart **[in]** first coordinate
	 * Computes the SPD at a given wavelenght _lambda_
	 * @return SPD value
	 */
	float interpolateSpectrumSamples(const float *lambda, const float *vals, int n, float l) {
		if(l <= lambda[0]) return vals[0];
		if(l >= lambda[n - 1]) return vals[n - 1];
		for(int i = 0; i < n - 1; i++)
			if(l >= lambda[i] && l <= lambda[i + 1]) {
				float t = (l - lambda[i]) / (lambda[i + 1] - lambda[i]);
				return ponos::lerp(t, vals[i], vals[i + 1]);
			}
		// TODO is it possible?
		std::cout << LOG_LOCATION << std::endl;
		return 0.f;
	}

	/* RGB spectrum implementation
	 */
	class RGBSpectrum : public CoefficientSpectrum<3> {
		public:
			RGBSpectrum(float v = 0.f) : CoefficientSpectrum<3>(v) {}
			RGBSpectrum(const CoefficientSpectrum<3> &v) : CoefficientSpectrum<3>(v) {}

			/* conversion
			 * @rgb **[out]**
			 */
			void toRGB(float *rgb) const {
				rgb[0] = c[0];
				rgb[1] = c[1];
				rgb[2] = c[2];
			}
			/* create
			 * @return RBG spectrum object
			 */
			const RGBSpectrum &toRGBSpectrum() const {
				return *this;
			}
			/* conversion
			 * @rgb **[in]**
			 * @type **[in | optional]** denotes wether the RGB values represents a surface reflectance or an illuminant.
			 * Converts from a given **rgb** values to a full SPD.
			 * @return
			 */
			static RGBSpectrum fromRGB(const float rgb[3], SpectrumType type = SpectrumType::REFLECTANCE) {
				RGBSpectrum s;
				s.c[0] = rgb[0];
				s.c[1] = rgb[1];
				s.c[2] = rgb[2];
				return s;
			}
			/* construct
			 * @lambda **[in]** array of lambda coordinates
			 * @v **[in]** array of values (one for each lambda coordinate)
			 * @n **[in]** number of coordinates
			 *
			 * Creates a piece-wise linear function to represent the SPD from the given set of samples.
			 * @return a RGBSpectrum object
			 */
			static RGBSpectrum fromSampled(const float *lambda, const float *v, int n) {
				// sort samples if unordered
				if(!std::is_sorted(&lambda[0], &lambda[n])) {
					std::vector<std::pair<float, float> > sortedSamples(n);
					for(int i = 0; i < n; i++) {
						sortedSamples[i].first = lambda[i];
						sortedSamples[i].second = v[i];
					};
					std::sort(sortedSamples.begin(), sortedSamples.end());
					std::vector<float> slambda(n);
					std::vector<float> sv(n);
					for(int i = 0; i < n; i++) {
						slambda[i] = sortedSamples[i].first;
						sv[i] = sortedSamples[i].second;
					}
					return fromSampled(&slambda[0], &sv[0], n);
				}
				float xyz[3] = {0, 0, 0};
				float yint = 0.f;
				for(int i = 0; i < nCIESamples; i++) {
					yint += CIE_Y[i];
					float val = interpolateSpectrumSamples(lambda, v, n, CIE_lambda[i]);
					xyz[0] += val * CIE_X[i];
					xyz[1] += val * CIE_Y[i];
					xyz[2] += val * CIE_Z[i];
				}
				xyz[0] /= yint;
				xyz[1] /= yint;
				xyz[2] /= yint;
				return fromXYZ(xyz);
			}

			static RGBSpectrum fromXYZ(const float xyz[3],
					SpectrumType type = SpectrumType::REFLECTANCE) {
				RGBSpectrum r;
				XYZToRGB(xyz, r.c);
				return r;
			}

			float y() const {
				const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
				return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
			}
	};

	/* spectrum implementation
	 * Represent a SPD with uniformly spaced samples over an wavelenght range.
	 */
	class SampledSpectrum : public CoefficientSpectrum<nSpectralSamples> {
		public:
			/* Constructor.
			 * @v **[in | optional]** constant value across all wavelenghts
			 */
			SampledSpectrum(float v = 0.f) {
				for(int i = 0; i < nSpectralSamples; i++)
					c[i] = v;
			}
			SampledSpectrum(const CoefficientSpectrum<nSpectralSamples> &v)
        : CoefficientSpectrum<nSpectralSamples>(v) {}
			SampledSpectrum(const RGBSpectrum &r, SpectrumType t) {
				float rgb[3];
				r.toRGB(rgb);
				*this = SampledSpectrum::fromRGB(rgb, t);
			}
			virtual ~SampledSpectrum() {}
			/* construct
			 * @lambda **[in]** array of lambda coordinates
			 * @v **[in]** array of values (one for each lambda coordinate)
			 * @n **[in]** number of coordinates
			 *
			 * Creates a piece-wise linear function to represent the SPD from the given set of samples.
			 * @return a SampledSpectrum object
			 */
			static SampledSpectrum fromSample(const float *lambda, const float *v, int n) {
				// sort samples if unordered
				if(!std::is_sorted(&lambda[0], &lambda[n])) {
					std::vector<std::pair<float, float> > sortedSamples(n);
					for(int i = 0; i < n; i++) {
						sortedSamples[i].first = lambda[i];
						sortedSamples[i].second = v[i];
					};
					std::sort(sortedSamples.begin(), sortedSamples.end());
					std::vector<float> slambda(n);
					std::vector<float> sv(n);
					for(int i = 0; i < n; i++) {
						slambda[i] = sortedSamples[i].first;
						sv[i] = sortedSamples[i].second;
					}
					return fromSample(&slambda[0], &sv[0], n);
				}
				SampledSpectrum r;
				// Compute averages
				for(int i = 0; i < nSpectralSamples; i++) {
					float lambda0 = ponos::lerp(static_cast<float>(i) / static_cast<float>(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
					float lambda1 = ponos::lerp(static_cast<float>(i + 1) / static_cast<float>(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
					r.c[i] = averageSpectrumSamples(lambda, v, n, lambda0, lambda1);
				}
				return r;
			}
			/* initialization
			 * computes XYZ matching curves
			 */
			static void init() {
				for(int i = 0; i < nSpectralSamples; i++) {
					float wl0 = ponos::lerp(static_cast<float>(i) / static_cast<float>(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
					float wl1 = ponos::lerp(static_cast<float>(i + 1) / static_cast<float>(nSpectralSamples), sampledLambdaStart, sampledLambdaEnd);
					X.c[i] = averageSpectrumSamples(CIE_lambda, CIE_X, nCIESamples, wl0, wl1);
					Y.c[i] = averageSpectrumSamples(CIE_lambda, CIE_Y, nCIESamples, wl0, wl1);
					Z.c[i] = averageSpectrumSamples(CIE_lambda, CIE_Z, nCIESamples, wl0, wl1);
					yint += Y.c[i];
				}
			}
			/* create
			 * @return RBG spectrum object
			 */
			RGBSpectrum toRGBSpectrum() const;
			/* conversion
			 * @xyz **[out]**
			 */
			void toXYZ(float xyz[3]) const {
				xyz[0] = xyz[1] = xyz[2] = 0.f;
				for(int i = 0; i < nSpectralSamples; i++) {
					xyz[0] += X.c[i] * c[i];
					xyz[1] += Y.c[i] * c[i];
					xyz[2] += Z.c[i] * c[i];
				}
				xyz[0] /= yint;
				xyz[1] /= yint;
				xyz[2] /= yint;
			}
			/* conversion
			 * @rgb **[out]**
			 */
			void toRGB(float rgb[3]) const {
				float xyz[3];
				toXYZ(xyz);
				XYZToRGB(xyz, rgb);
			}
			/* get
			 * The y coordinate of XYZ color is closely related to _luminance_
			 */
			float y() const {
				float yy = 0.f;
				for(int i = 0; i < nSpectralSamples; i++)
					yy += Y.c[i] * c[i];
				return yy / yint;
			}
			/* conversion
			 * @rgb **[in]**
			 * @type **[in]** denotes wether the RGB values represents a surface reflectance or an illuminant.
			 * Converts from a given **rgb** values to a full SPD.
			 * @return
			 */
			static SampledSpectrum fromRGB(const float rgb[3], SpectrumType type);
			/* conversion
			 * @xyz **[in]**
			 * @type **[in | optional]** denotes wether the RGB values represents a surface reflectance or an illuminant.
			 * Converts from a given **xyz** values to a full SPD.
			 * @return
			 */
			static SampledSpectrum fromXYZ(const float xyz[3], SpectrumType type = SpectrumType::REFLECTANCE) {
				float rgb[3];
				XYZToRGB(xyz, rgb);
				return fromRGB(rgb, type);
			}

		private:
			static SampledSpectrum X, Y, Z;
			static float yint;
			static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
			static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
			static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
			static SampledSpectrum rgbRefl2SpectBlue;
			static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
			static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
			static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
			static SampledSpectrum rgbIllum2SpectBlue;
	};

	//typedef RGBSpectrum Spectrum;
	typedef SampledSpectrum Spectrum;

} // helios namespace

#endif // HELIOS_CORE_SPECTRUM_H

