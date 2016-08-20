#ifndef HELIOS_GEOMETRY_H_RAY_H
#define HELIOS_GEOMETRY_H_RAY_H

#include <ponos.h>

namespace helios {

	/* Implements <ponos::Ray3>.
	 * Has additional information related to the ray tracing algorithm.
	 */
  class HRay : public ponos::Ray3 {
  public:
    HRay();
		/* Constructor.
		 * @origin
		 * @direction
		 * @start parametric coordinate start
		 * @end parametric coordinate end (how far this ray can go)
		 * @t parametric coordinate
		 * @d depth of recursion
		 */
    explicit HRay(const ponos::Point3& origin, const ponos::Vector3& direction,
        float start = 0.f, float end = INFINITY, float t = 0.f, int d = 0);
		/* Constructor.
		 * @origin
		 * @direction
		 * @parent
		 * @start parametric coordinate start
		 * @end parametric coordinate end (how far this ray can go)
		 */
		explicit HRay(const ponos::Point3& origin, const ponos::Vector3& direction, const HRay& parent,
        float start, float end = INFINITY);

		mutable float min_t, max_t;
		float time;
		int depth;
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
			explicit RayDifferential(const ponos::Point3& origin,
					const ponos::Vector3& direction, float start = 0.f, float end = INFINITY,
					float t = 0.f, int d = 0);
			/* Constructor.
			 * @origin
			 * @direction
			 * @parent
			 * @start parametric coordinate start
			 * @end parametric coordinate end (how far this ray can go)
			 */
			explicit RayDifferential(const ponos::Point3& origin,
					const ponos::Vector3& direction, const HRay& parent, float start,
					float end = INFINITY);

			explicit RayDifferential(const HRay &ray)
				: HRay(ray) { hasDifferentials = false; }

			/* Updates the differential rays
			 * @s smaple spacing
			 *
			 * Updates the differential rays for a sample spacing <s>.
			 */
			void scaleDifferentials(float s);
			bool hasDifferentials;
			ponos::Point3 rxOrigin, ryOrigin;
			ponos::vec3 rxDirection, ryDirection;
	};

} // helios namespace

#endif
