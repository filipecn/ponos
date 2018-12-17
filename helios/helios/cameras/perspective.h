#ifndef HELIOS_CAMERAS_PERSPECTIVE_H
#define HELIOS_CAMERAS_PERSPECTIVE_H

#include "core/camera.h"

namespace helios {

	/* Perspective Camera model.
	 * Camera based on the perspective projection transformation.
	 */
	class PerspectiveCamera : public ProjectiveCamera {
  	public:
			/* Constructor
			 * @cam2world camera to world transform.
			 * @proj
			 * @screenWindow
			 * @sopen
			 * @sclose
			 * @lensr
			 * @focald
			 * @fv
			 * @film
			 */
			PerspectiveCamera(const AnimatedTransform &cam2world,
					const float screenWindow[4],
					float sopen, float sclose, float lensr,
					float focald, float fv, Film *f);

			/* @inherit */
			virtual float generateRay(const CameraSample &sample, HRay *ray) const override;
			/* @inherit */
			virtual float generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const override;

			virtual ~PerspectiveCamera() {}
		private:
			ponos::vec3 dxCamera, dyCamera;
	};

} // helios namespace

#endif // HELIOS_CAMERAS_PERSPECTIVE_H

