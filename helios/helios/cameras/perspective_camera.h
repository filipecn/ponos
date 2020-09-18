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
   * \param cam2world camera to world transform
   * \param screenWindow screen space extent of the image
   * \param sopen shutter open time
   * \param sclose shutter close time
   * \param lensr lens radius
   * \param focald focal distance
   * \param fov field of view
   * \param film
   */
  PerspectiveCamera(const AnimatedTransform &cam2world,
                    const bounds2 &screenWindow, real_t sopen, real_t sclose,
                    real_t lensr, real_t focald, real_t fov,
                    Film *f /*, const Medium *medium*/);
  virtual ~PerspectiveCamera() {}
  virtual real_t generateRay(const CameraSample &sample,
                             HRay *ray) const override;
  virtual real_t generateRayDifferential(const CameraSample &sample,
                                         RayDifferential *rd) const override;

private:
  ponos::vec3 dxCamera, dyCamera;
};

} // namespace helios

#endif // HELIOS_CAMERAS_PERSPECTIVE_H
