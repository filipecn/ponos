#include "core/camera.h"

namespace helios {

	Camera::Camera(const AnimatedTransform &cam2world, float sopen,
			float sclose, Film *film)
		: cameraToWorld(cam2world),
		shutterOpen(sopen),
		shutterClose(sclose),
		film(film) {}

	float Camera::generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const {
		float wt = generateRay(sample, rd);
		// find ray after shifting one pixel in the x direction
		CameraSample sshift = sample;
		++(sshift.imageX);
		HRay rx;
		float wtx = generateRay(sshift, &rx);
		rd->rxOrigin = rx.o;
		rd->rxDirection = rx.d;
		// find ray after shifting one pixel in the y direction
		CameraSample sshift_y = sample;
		++(sshift_y.imageY);
		HRay ry;
		float wty = generateRay(sshift_y, &ry);
		rd->ryOrigin = ry.o;
		rd->ryDirection = ry.d;
		if(wtx == 0.f || wty == 0.f)
			return 0.f;
		rd->hasDifferentials = true;
		return wt;
	}

	ProjectiveCamera::ProjectiveCamera(const AnimatedTransform &cam2world,
					const ponos::Transform &proj, const float screenWindow[4],
					float sopen, float sclose, float lensr,
					float focald, Film *f)
		: Camera(cam2world, sopen, sclose, f) {
			// initialize depth of field parameters
			lensRadius = lensr;
			focalDistance = focald;
			// compute projective camera transformations
			cameraToScreen = proj;
			// compute projective camera screen transformations
			screenToRaster = ponos::scale(float(film->xResolution),
																		float(film->yResolution), 1.f) *
				ponos::scale(1.f / (screenWindow[1] - screenWindow[0]),
										 1.f / (screenWindow[2] - screenWindow[3]), 1.f) *
				ponos::translate(ponos::vec3(-screenWindow[0], -screenWindow[3], 0.f));
			rasterToScreen = ponos::inverse(screenToRaster);
			rasterToCamera = ponos::inverse(cameraToScreen) * rasterToScreen;
		}

} // helios namespace
