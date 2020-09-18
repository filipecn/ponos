#ifndef HELIOS_CAMERAS_ORTHOGRAPHIC_H
#define HELIOS_CAMERAS_ORTHOGRAPHIC_H

#include <helios/core/camera.h>
#include <helios/geometry/animated_transform.h>

#include <ponos.h>

namespace helios {

/* Orthographic Camera model.
 * Camera based on the orthographic projection transformation.
 */
class OrthoCamera : public ProjectiveCamera {
public:
  /* 
   * \param cam2world camera to world transform
   * \param screenWindow screen space extent of the image
   * \param sopen shutter open time
   * \param sclose shutter close time
   * \param lensr
   * \param focald
   * \param film
   */
  OrthoCamera(const AnimatedTransform &cam2world, const bounds2 screenWindow,
              float sopen, float sclose, float lensr, float focald, Film *f,
              /*const Medium * medium*/);

  virtual ~OrthoCamera() {}
  virtual real_t generateRay(const CameraSample &sample,
                            HRay *ray) const override;
  virtual real_t generateRayDifferential(const CameraSample &sample,
                                        RayDifferential *rd) const override;

private:
  ponos::vec3 dxCamera, dyCamera;
};

} // namespace helios

#endif // HELIOS_CAMERAS_ORTHOGRAPHIC_H
