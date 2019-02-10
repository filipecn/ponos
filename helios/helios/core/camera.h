/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HELIOS_CORE_CAMERA_H
#define HELIOS_CORE_CAMERA_H

#include <helios/geometry/animated_transform.h>
#include <ponos/geometry/point.h>

namespace helios {

/* Sample information.
 * Stores the values needed for generating camera rays.
 */
struct CameraSample {
  ponos::point2f pFilm; //!< point on the film to which the ray carries radiance
  ponos::point2f pLens; //!< point on the lens the ray passes through
  real_t time;          //!< time at wich the ray should sample the scene
};

/* Camera base class.
 * Defines the interface for camera implementations.
 */
class Camera {
public:
  /* \param cam2world camera to world transform.
   * \param shutterOpen shutter open time
   * \param shutterClose shutter close time
   * @film
   */
  Camera(const AnimatedTransform &cam2world, real_t shutterOpen,
         real_t shutterClose, Film *film /*, const Medium* medium*/);
  virtual ~Camera() {}
  /* Generates ray for a given image sample.
   * \param sample
   * \param ray
   * the camera must normalize the direction of <ray>.
   * \return a weight of how much light is arriving at the film plane.
   */
  virtual real_t generateRay(const CameraSample &sample, HRay *ray) const = 0;
  /* .
   * @sample
   * @rd
   *
   * Also computes the corresponding rays for pixels shifted one pixel in the x
   * and y directions on the plane.
   * @return a weight of how much light is arriving at the film plane.
   */
  virtual real_t generateRayDifferential(const CameraSample &sample,
                                        RayDifferential *rd) const;
  AnimatedTransform cameraToWorld;
  const real_t shutterOpen, shutterClose;
  Film *film;
};

/* Projective Camera model.
 * Base for models that use a 4x4 projective transformation matrix.
 */
class ProjectiveCamera : public Camera {
public:
  /*
   * \param cam2world camera to world transform
   * \param cam2scr camera to screen space transform
   * \param screenWindow screen space extent of the image
   * \param sopen shutter open time
   * \param sclose shutter close time
   * \param lensr lens raidius
   * \param focald focal distance
   * \param film
   */
  ProjectiveCamera(const AnimatedTransform &cam2world,
                   const HTransform &cam2scr, const bounds2 &screenWindow,
                   real_t sopen, real_t sclose, real_t lensr, real_t focald,
                   Film *f /*, const Medium *medium*/);

protected:
  HTransform cameraToScreen, rasterToCamera;
  HTransform screenToRaster, rasterToScreen;
  real_t focalDistance, lensRadius;
};

} // namespace helios

#endif // HELIOS_CORE_CAMERA_H
