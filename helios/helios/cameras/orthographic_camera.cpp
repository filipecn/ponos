#include "orthographic_camera.h"

namespace helios {

OrthoCamera::OrthoCamera(const AnimatedTransform &cam2world,
                         const bounds2 screenWindow, float sopen, float sclose,
                         float lensr, float focald, Film *f)
    : ProjectiveCamera(cam2world, ponos::orthographic(0., 1.), screenWindow,
                       sopen, sclose, lensr, focald, f) {
  // compute differential changes in origin for ortho camera rays
  dxCamera = rasterToCamera(ponos::vec3(1, 0, 0));
  dyCamera = rasterToCamera(ponos::vec3(0, 1, 0));
}

real_t OrthoCamera::generateRay(const CameraSample &sample, HRay *ray) const {
  // generate raster and camera sample positions
  ponos::Point3 pFilm(sample.pFilm.x, sample.pFilm.y, 0);
  ponos::Point3 pCamera = rasterToCamera(pFilm);
  *ray = HRay(pCamera, ponos::vec3(0, 0, 1));
  // modify ray for depth of field
  if (lensRadius > 0.) {
    // TODO pg 315
    // sample point on lens
    // compute point on plane of focus
    // update ray for effect of lens
  }
  ray->time = ponos::lerp(sample.time, shutterOpen, shutterClose);
  // TODO ENABLE ray->medium = medium;
  cameraToWorld(*ray, ray);
  return 1;
}

real_t OrthoCamera::generateRayDifferential(const CameraSample &sample,
                                            RayDifferential *rd) const {
  // compute main orthographic viewing ray
  // compute ray differentials for krthografic camera
  if (lensRadius > 0) {
    // caompute orthographic camera ray differentials accounting for lens
  } else {
    rd->rxOrigin = rd->o + dxCamera;
    rd->ryOrigin = rd->o + dyCamera;
    rd->rxDirection = rd->ryDirection = rd->d;
  }
  rd->time = ponos::lerp(sample.time, shutterOpen, shutterClose);
  rd->hasDifferentials = true;
  // TODO ENABLE ray->medium = medium;
  cameraToWorld(*rd, rd);
  return 1;
}

} // namespace helios
