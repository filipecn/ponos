#ifndef HELIOS_CORE_FILTER_H
#define HELIOS_CORE_FILTER_H

#include <ponos/geometry/vector.h>

namespace helios {

/// Base class for implementation of various types of filter functions.
class Filter {
public:
  ///  The components of **radius** define points in the axis where the function
  ///  is zero beyond. These points go to each direction, the overall extent
  ///  (its _support_) in each direction is **twice** those values.
  /// \param radius filter extents
  Filter(const ponos::vec2 &radius);
  /// Evaluates filter functiona at **p**
  /// \param p sample point relative to the center of the filter
  /// \return real_t filter's value
  virtual real_t evaluate(const ponos::point2 &p) const = 0;

  const ponos::vec2 radius;    //!< filter's radius of support
  const ponos::vec2 invRadius; //!< reciprocal of radius
};

/// Equally weights all samples within a square region of the image.
/// Can cause postaliasing even when the original image's frequencies
/// respect the Nyquist limit.
class BoxFilter : public Filter {
  BoxFilter(const ponos::vec2 &radius);
  real_t evaluate(const ponos::point2 &p) const override;
};
} // namespace helios

#endif // HELIOS_CORE_FILTER_H
