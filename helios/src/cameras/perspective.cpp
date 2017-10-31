#include "cameras/perspective.h"

#include "geometry/animated_transform.h"

namespace helios {

	PerspectiveCamera::PerspectiveCamera(const AnimatedTransform &cam2world,
					const float screenWindow[4],
					float sopen, float sclose, float lensr,
					float focald, float fv, Film *f)
		: ProjectiveCamera(cam2world, ponos::perspective(fv, 1e-2f, 1000.f),
				screenWindow, sopen, sclose, lensr, focald, f) {
			// compute differential changes in origin for perspective camera rays
			dxCamera = rasterToCamera(ponos::Point3(1, 0, 0)) - rasterToCamera(ponos::Point3(0, 0, 0));
			dyCamera = rasterToCamera(ponos::Point3(0, 1, 0)) - rasterToCamera(ponos::Point3(0, 0, 0));
		}

		float PerspectiveCamera::generateRay(const CameraSample &sample, HRay *ray) const {
			// generate raster and camera samples
			ponos::Point3 pRas(sample.imageX, sample.imageY, 0);
			ponos::Point3 pCamera;
			rasterToCamera(pRas, &pCamera);
			*ray = HRay(ponos::Point3(0, 0, 0), ponos::vec3(pCamera), 0.f, INFINITY);
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

		float PerspectiveCamera::generateRayDifferential(const CameraSample &sample, RayDifferential *rd) const {
			// generate raster and camera samples
			ponos::Point3 pRas(sample.imageX, sample.imageY, 0);
			ponos::Point3 pCamera;
			rasterToCamera(pRas, &pCamera);
			// compute offset rays
			rd->rxOrigin = rd->ryOrigin = rd->o;
			rd->rxDirection = ponos::normalize(ponos::vec3(pCamera) + dxCamera);
			rd->ryDirection = ponos::normalize(ponos::vec3(pCamera) + dyCamera);
			rd->hasDifferentials = true;
			cameraToWorld(*rd, rd);
			return 1.f;
		}

} // helios namespace
