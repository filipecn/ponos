#ifndef HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H
#define HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H

#include <helios/core/sampler.h>

namespace helios {

/// Divides the image into rectangular regions and generates  a single sample
/// inside each region (_stratum_).
class StratifiedSampler : public PixelSampler {
public:
  ///
  /// \param xPixelSamples
  /// \param yPixelSamples
  /// \param jitterSamples
  /// \param nSampledDimensions
  StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitterSamples,
                    int nSampledDimensions);
  void startPixel(const ponos::point2i &p) override;

private:
  const int xPixelSamples;
  const int yPixelSamples;
  const bool jitterSamples;
};

/// Divides the image into rectangular regions and generates  a single sample
/// inside each region (_stratum_).
// class StratifiedSampler : public Sampler {
// public:
//   /* Constructor.
//    * \param xstart **[in]** first pixel coordinate on X axis
//    * \param xend **[in]** (last + 1) pixel coordinate on X axis
//    * \param ystart **[in]** first pixel coordinate on Y axis
//    * \param yend **[in]**(last + 1) pixel coordinate on Y axis
//    * \param xs **[in]** number of _strata_ in X axis
//    * \param ys **[in]** number of _strata_ in Y axis
//    * \param jitter **[in]** if true, the samples are jittered
//    * \param sopen **[in]** camera's shutter open time
//    * \param sclose **[in]** camera's shutter close time
//    */
//   StratifiedSampler(int xstart, int xend, int ystart, int yend, int xs, int
//   ys,
//                     bool jitter, float sopen, float sclose);
//   virtual ~StratifiedSampler() {}
//   /* @inherit */
//   int getMoreSamples(Sample *samples, ponos::RNG &rng) override;
//   /* @inherit */
//   int maximumSampleCount() override { return xPixelSamples * yPixelSamples; }
//   /* @inherit
//    * @return null if input is degenerated
//    **/
//   Sampler *getSubSampler(int num, int count) override;
//   /* @inherit */
//   int roundSize(int size) const override { return size; }

// private:
//   int xPixelSamples, yPixelSamples;
//   bool jitterSamples;
//   int xPos, yPos;
//   float *buffer;
// };

} // namespace helios

#endif // HELIOS_SAMPLERS_STRATIFIED_SAMPLER_H
