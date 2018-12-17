#ifndef HELIOS_GEOMETRY_H_RAY_H
#define HELIOS_GEOMETRY_H_RAY_H

#include <ponos/ponos.h>

namespace helios {

/* Implements <ponos::Ray3>.
 * Has additional information related to the ray tracing algorithm.
 */
class HRay {
public:
  HRay();
  /// \param origin
  /// \param direction
  /// \param tMax max parametric corrdinate allowed for this ray
  /// \param time current parametric coordinate (time value)
  HRay(const ponos::point3f &origin, const ponos::vec3f &direction,
       real_t tMax = ponos::Constants::real_infinity,
       real_t time = 0.f /*, const Medium* medium = nullptr*/);
  ponos::point3f operator()(real_t t) const;

  ponos::point3f o; //!< ray's origin
  ponos::vec3f d; //!< ray's direction
  mutable real_t max_t; //!< max parametric distance allowed
  //  const Medium* medium; //!< medium containing **o**
  real_t time; //!< current parametric distance
};

/* Subclass of <helios::Hray>.
 * Has 2 auxiliary rays to help algorithms like antialiasing.
 */
class RayDifferential : public HRay {
public:
  RayDifferential() { hasDifferentials = false; }
  /* Constructor.
   * @origin
   * @direction
   * @start parametric coordinate start
   * @end parametric coordinate end (how far this ray can go)
   * @t parametric coordinate
   * @d depth of recursion
   */
  explicit RayDifferential(const ponos::point3 &origin,
                           const ponos::vec3 &direction, float start = 0.f,
                           float end = INFINITY, float t = 0.f, int d = 0);
  /* Constructor.
   * @origin
   * @direction
   * @parent
   * @start parametric coordinate start
   * @end parametric coordinate end (how far this ray can go)
   */
  explicit RayDifferential(const ponos::point3 &origin,
                           const ponos::vec3 &direction, const HRay &parent,
                           float start, float end = INFINITY);

  explicit RayDifferential(const HRay &ray) : HRay(ray) {
    hasDifferentials = false;
  }

  /* Updates the differential rays
   * @s smaple spacing
   *
   * Updates the differential rays for a sample spacing <s>.
   */
  void scaleDifferentials(float s);
  bool hasDifferentials;
  ponos::point3 rxOrigin, ryOrigin;
  ponos::vec3 rxDirection, ryDirection;
};

} // namespace helios

#endif
