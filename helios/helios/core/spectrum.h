#ifndef HELIOS_CORE_SPECTRUM_H
#define HELIOS_CORE_SPECTRUM_H

#include <ponos/ponos.h>

#include <algorithm>
#include <cmath>
#include <map>

namespace helios {

static const int nCIESamples = 471;
extern const real_t CIE_X[nCIESamples];
extern const real_t CIE_Y[nCIESamples];
extern const real_t CIE_Z[nCIESamples];
extern const real_t CIE_lambda[nCIESamples];
static const real_t CIE_Y_integral = 106.856895;

static const int nRGB2SpectSamples = 32;
extern const real_t RGB2SpectLambda[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectWhite[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectCyan[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectMagenta[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectYellow[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectRed[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectGreen[nRGB2SpectSamples];
extern const real_t RGBRefl2SpectBlue[nRGB2SpectSamples];

extern const real_t RGBIllum2SpectWhite[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectCyan[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectMagenta[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectYellow[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectRed[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectGreen[nRGB2SpectSamples];
extern const real_t RGBIllum2SpectBlue[nRGB2SpectSamples];

/// \param xyz **[in]** xyz values
/// \param rgb **[out]** returned rgb values
inline void XYZToRGB(const real_t xyz[3], real_t rgb[3]) {
  rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
  rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
  rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}
/// \param rgb **[in]** rgb values
/// \param xyz **[out]** returned xyz values
inline void RGBToXYZ(const real_t rgb[3], real_t xyz[3]) {
  xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
  xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
  xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

/* base class
 * Represents a spectrum by a particular number of samples (given by
 * **nSpectrumSamples** parameter) of the SPD(_spectral power distribution_).
 */
template <int nSpectrumSamples> class CoefficientSpectrum {
public:
  /// \param v **[in | optional]** constant value across all wave lengths
  CoefficientSpectrum(float v = 0.f);
  virtual ~CoefficientSpectrum() = default;

  real_t &operator[](int i);
  CoefficientSpectrum &operator+=(const real_t &f);
  CoefficientSpectrum operator+(const real_t &f) const;
  CoefficientSpectrum &operator+=(const CoefficientSpectrum &s2);
  CoefficientSpectrum operator+(const CoefficientSpectrum &s2) const;
  CoefficientSpectrum operator-(const CoefficientSpectrum &s2) const;
  CoefficientSpectrum &operator*=(const CoefficientSpectrum &s2);
  CoefficientSpectrum operator*(const CoefficientSpectrum &s2) const;
  friend inline CoefficientSpectrum operator*(const real_t &f,
                                              const CoefficientSpectrum &s) {
    return s * f;
  }
  CoefficientSpectrum &operator*=(const real_t &f);
  CoefficientSpectrum operator*(const real_t &f) const;
  CoefficientSpectrum &operator/=(const CoefficientSpectrum &s2);
  CoefficientSpectrum operator/(const CoefficientSpectrum &s2) const;
  CoefficientSpectrum &operator/=(const real_t &f);
  CoefficientSpectrum operator/(const real_t &f) const;
  /// Useful to ray tracing algorithms to avoid the computation of reflected
  /// rays of zero reflectance surfaces.
  /// \return true
  /// \return false
  bool isBlack() const;
  /// \param l **[in | optional]** low (default = 0)
  /// \param h **[in | optional]** high (default = INFINITY)
  /// \return CoefficientSpectrum the spectrum with values clamped to be in
  /// **[l, h]**
  CoefficientSpectrum clamp(real_t l = 0,
                            real_t h = ponos::Constants::real_infinity) const;
  bool hasNaNs() const;

  friend CoefficientSpectrum sqrt(const CoefficientSpectrum &s) {
    CoefficientSpectrum r;
    for (int i = 0; i < nSpectrumSamples; i++)
      r.c[i] = sqrtf(s.c[i]);
    return r;
  }

  static const int nSamples = nSpectrumSamples;

protected:
  real_t c[nSpectrumSamples];
};

#include "spectrum.inl"

static const int sampledLambdaStart = 400;
static const int sampledLambdaEnd = 700;
static const int nSpectralSamples = 60;

enum class SpectrumType { REFLECTANCE, ILLUMINANT };

/// Computes the average value of a list of samples.
/// \param lambda **[in]** coordinates
/// \param vals **[in]** values on coordinates
/// \param n **[in]** number of samples
/// \param lambdaStart **[in]** first coordinate
/// \param lambdaEnd **[in]** last coordinate
/// \return real_t
real_t averageSpectrumSamples(const real_t *lambda, const real_t *vals, int n,
                              real_t lambdaStart, real_t lambdaEnd);
/// Computes the SPD at a given wavelength _lambda_
/// \param lambda **[in]** coordinates
/// \param vals **[in]** values on coordinates
/// \param n **[in]** number of samples
/// \param lambdaStart **[in]** first coordinate
/// \return SPD value
real_t interpolateSpectrumSamples(const real_t *lambda, const real_t *vals,
                                  int n, real_t l);
/// RGB spectrum implementation
class RGBSpectrum : public CoefficientSpectrum<3> {
public:
  RGBSpectrum(real_t v = 0.f);
  RGBSpectrum(const CoefficientSpectrum<3> &v);
  /// \param rgb **[out]**
  void toRGB(real_t *rgb) const;
  /// return RBG spectrum object
  const RGBSpectrum &toRGBSpectrum() const;
  /// \param rgb **[in]**
  /// \param type **[in | optional]** denotes wether the RGB values represents a
  /// surface reflectance or an illuminant. Converts from a given **rgb** values
  /// to a full SPD.
  /// \return RGBSpectrum
  static RGBSpectrum fromRGB(const real_t rgb[3],
                             SpectrumType type = SpectrumType::REFLECTANCE);
  /// Creates a piece-wise linear function to represent the SPD from the given
  /// set of samples.
  /// \param lambda **[in]** array of lambda coordinates
  /// \param v **[in]** array of values (one for each lambda coordinate)
  /// \param n **[in]** number of coordinates
  /// \return a RGBSpectrum object
  static RGBSpectrum fromSampled(const real_t *lambda, const real_t *v, int n);

  static RGBSpectrum fromXYZ(const real_t xyz[3],
                             SpectrumType type = SpectrumType::REFLECTANCE);
  real_t y() const;
};

/// Represent a SPD with uniformly spaced samples over an wavelength range.
/// The wavelengths covers from 400 nm to 700 nm
class SampledSpectrum : public CoefficientSpectrum<nSpectralSamples> {
public:
  /* Constructor.
   * @v **[in | optional]** constant value across all wavelengths
   */
  SampledSpectrum(real_t v = 0.f);
  SampledSpectrum(const CoefficientSpectrum<nSpectralSamples> &v);
  SampledSpectrum(const RGBSpectrum &r, SpectrumType t);
  virtual ~SampledSpectrum() = default;
  /// Creates a piece-wise linear function to represent the SPD from the given
  /// set of samples.
  /// \param lambda **[in]** array of lambda coordinates
  /// \param v **[in]** array of values (one for each lambda coordinate)
  /// \param n **[in]** number of coordinates
  /// \return SampledSpectrum
  static SampledSpectrum fromSamples(const real_t *lambda, const real_t *v,
                                     int n);
  /// computes XYZ matching curves
  static void init();
  /// \return RBG spectrum object
  RGBSpectrum toRGBSpectrum() const;
  /// \param xyz **[out]**
  void toXYZ(float xyz[3]) const;
  /// \param rgb **[out]**
  void toRGB(float rgb[3]) const;
  /// \return real_t  The y coordinate of XYZ color is closely related to
  /// _luminance_
  real_t y() const;
  /// Converts from a given **rgb** values to a full SPD.
  /// \param rgb **[in]**
  /// \param type **[in]** denotes wether the RGB values represents a surface
  /// reflectance or an illuminant.
  /// \return SampledSpectrum
  static SampledSpectrum fromRGB(const float rgb[3], SpectrumType type);
  /// surface reflectance or an illuminant. Converts from a given **xyz**
  /// values to a full SPD.
  /// \param xyz **[in]**
  /// \param type **[in | optional]** denotes wether the RGB values represents a
  /// \return SampledSpectrum
  static SampledSpectrum fromXYZ(const float xyz[3],
                                 SpectrumType type = SpectrumType::REFLECTANCE);

private:
  static SampledSpectrum X, Y, Z;
  static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
  static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
  static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
  static SampledSpectrum rgbRefl2SpectBlue;
  static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
  static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
  static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
  static SampledSpectrum rgbIllum2SpectBlue;
};

// typedef RGBSpectrum Spectrum;
typedef SampledSpectrum Spectrum;

Spectrum lerp(real_t t, const Spectrum &a, const Spectrum &b);

} // namespace helios

#endif // HELIOS_CORE_SPECTRUM_H
