#ifndef HELIOS_CAMERAS_ORTHOGRAPHIC_H
#define HELIOS_CAMERAS_ORTHOGRAPHIC_H

#include "core/camera.h"
#include "geometry/animated_transform.h"

#include <ponos.h>

namespace helios {

	/* Orthographic Camera model.
	 * Camera based on the orthographic projection transformation.
	 */
	class OrthoCamera : public ProjectiveCamera {
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
			OrthoCamera(const AnimatedTransform &cam2world,
					const float screenWindow[4],
					float sopen, float sclose, float lensr,
					float focald, Film *f);

			virtual ~OrthoCamera() {}
			/* @inherit */
			virtual float generateRay(const CameraSample &sample, HRay *ray) const override;
			/* @inherit */
			virtual float generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const override;

		private:
			ponos::vec3 dxCamera, dyCamera;
	};

} // helios namespace

#endif // HELIOS_CAMERAS_ORTHOGRAPHIC_H

