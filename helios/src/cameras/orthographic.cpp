#include "cameras/orthographic.h"

namespace helios {

	OrthoCamera::OrthoCamera(const AnimatedTransform &cam2world,
			const float screenWindow[4],
			float sopen, float sclose, float lensr,
			float focald, Film *f)
		: ProjectiveCamera(cam2world, ponos::orthographic(0., 1.), screenWindow, sopen, sclose, lensr, focald, f) {
			// compute differential changes in origin for ortho camera rays
			dxCamera = rasterToCamera(ponos::vec3(1, 0, 0));
			dyCamera = rasterToCamera(ponos::vec3(0, 1, 0));
		}

		float OrthoCamera::generateRay(const CameraSample &sample, HRay *ray) const {
			// generate raster and camera samples
			ponos::Point3 pRas(sample.imageX, sample.imageY, 0);
			ponos::Point3 pCamera;
			rasterToCamera(pRas, &pCamera);
			*ray = HRay(pCamera, ponos::vec3(0, 0, 1), 0.f, INFINITY);
			// modify ray for depth of field
			if(lensRadius > 0.) {
				// TODO pg 315
				// sample point on lens
				// compute point on plane of focus
				// update ray for effect of lens
			}
			ray->time = ponos::lerp(sample.time, shutterOpen, shutterClose);
			cameraToWorld(*ray, ray);
			return 1.f;
		}

		float OrthoCamera::generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const {
			// compute main orthographic viewing ray
			rd->rxOrigin = rd->o + dxCamera;
			rd->ryOrigin = rd->o + dyCamera;
			rd->rxDirection = rd->ryDirection = rd->d;
			rd->hasDifferentials = true;
			cameraToWorld(*rd, rd);
			return 1.f;
		}
} // helios namespace
