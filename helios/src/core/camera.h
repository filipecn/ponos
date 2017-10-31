#ifndef HELIOS_CORE_CAMERA_H
#define HELIOS_CORE_CAMERA_H

#include "core/film.h"
#include "core/sampler.h"
#include "geometry/animated_transform.h"
#include "geometry/h_ray.h"

#include <ponos.h>

namespace helios {

	/* Camera base class.
	 * Defines the interface for camera implementations.
	 */
	class Camera {
		public:
			/* Constructor
			 * @cam2world camera to world transform.
			 * @sopen
			 * @sclose
			 * @film
			 */
			Camera(const AnimatedTransform &cam2world, float sopen, float sclose, Film *film);
			virtual ~Camera() {}
			/* Generates ray for a given image sample.
			 * @sample
			 * @ray
			 *
			 * the camera must normalize the direction of <ray>.
			 *
			 * @return a weight of how much light is arriving at the film plane.
			 */
			virtual float generateRay(const CameraSample &sample, HRay *ray) const = 0;
			/* .
			 * @sample
			 * @rd
			 *
			 * Also computes the corresponding rays for pixels shifted one pixel in the x and y directions on the plane.
			 * @return a weight of how much light is arriving at the film plane.
			 */
			virtual float generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const;
			AnimatedTransform cameraToWorld;
			const float shutterOpen, shutterClose;
			Film *film;

		protected:
			float lensRadius, focalDistance;
	};

	/* Projective Camera model.
	 * Base for models that use a 4x4 projective transformation matrix.
	 */
	class ProjectiveCamera : public Camera {
		public:
			/* Constructor
			 * @cam2world camera to world transform.
			 * @proj
			 * @screenWindow
			 * @sopen
			 * @sclose
			 * @lensr
			 * @focald
			 * @film
			 */
			ProjectiveCamera(const AnimatedTransform &cam2world,
					const ponos::Transform &proj, const float screenWindow[4],
					float sopen, float sclose, float lensr,
					float focald, Film *f);

		protected:
			ponos::Transform cameraToScreen, rasterToCamera;
			ponos::Transform screenToRaster, rasterToScreen;
			float focalDistance, lensRadius;
	};

} // helios namespace

#endif // HELIOS_CORE_CAMERA_H

