#include "cameras/perspective.h"

#include "geometry/animated_transform.h"

namespace helios {

PerspectiveCamera::PerspectiveCamera(const AnimatedTransform &cam2world,
                                     const bounds2 &screenWindow, real_t sopen,
                                     real_t sclose, real_t lensr, real_t focald,
                                     real_t fv, Film *f)
    : ProjectiveCamera(cam2world, ponos::perspective(fv, 1e-2f, 1000.f),
                       screenWindow, sopen, sclose, lensr, focald, f) {
  // compute differential changes in origin for perspective camera rays
  dxCamera = rasterToCamera(ponos::Point3(1, 0, 0)) -
             rasterToCamera(ponos::Point3(0, 0, 0));
  dyCamera = rasterToCamera(ponos::Point3(0, 1, 0)) -
             rasterToCamera(ponos::Point3(0, 0, 0));
  // compute image plane bounds at z = 1 TODO 951
}

real_t PerspectiveCamera::generateRay(const CameraSample &sample,
                                      HRay *ray) const {
  // generate raster and camera samples
  ponos::Point3 pRas(sample.pFilm.x, sample.pFilm.y, 0);
  ponos::Point3 pCamera;
  rasterToCamera(pRas, &pCamera);
  *ray = HRay(ponos::Point3(0, 0, 0), ponos::normalize(ponos::vec3(pCamera)));
  // modify ray for depth of field
  if (lensRadius > 0.) {
    // TODO pg 315
    // sample point on lens
    // compute point on plane of focus
    // update ray for effect of lens
  }
  ray->time = ponos::lerp(sample.time, shutterOpen, shutterClose);
  // TODO ray->medium = medium;
  cameraToWorld(*ray, ray);
  return 1;
}

real_t PerspectiveCamera::generateRayDifferential(const CameraSample &sample,
                                                  RayDifferential *rd) const {
  // generate raster and camera samples
  ponos::Point3 pRas(sample.pFilm.x, sample.pFilm.y, 0);
  ponos::Point3 pCamera;
  rasterToCamera(pRas, &pCamera);
  // compute offset rays
  if (lensRadius > 0) {
  } else {
    rd->rxOrigin = rd->ryOrigin = rd->o;
    rd->rxDirection = ponos::normalize(ponos::vec3(pCamera) + dxCamera);
    rd->ryDirection = ponos::normalize(ponos::vec3(pCamera) + dyCamera);
  }
  rd->hasDifferentials = true;
  cameraToWorld(*rd, rd);
  return 1;
}

} // namespace helios
