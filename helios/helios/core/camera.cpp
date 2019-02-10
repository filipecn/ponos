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

#include "camera.h"

namespace helios {

Camera::Camera(const AnimatedTransform &cam2world, real_t sopen, real_t sclose,
               Film *film)
    : cameraToWorld(cam2world), shutterOpen(sopen), shutterClose(sclose),
      film(film) {}

real_t Camera::generateRayDifferential(const CameraSample &sample,
                                       RayDifferential *rd) const {
  real_t wt = generateRay(sample, rd);
  // find ray after shifting one pixel in the x direction
  CameraSample sshift = sample;
  ++(sshift.pFilm.x);
  HRay rx;
  real_t wtx = generateRay(sshift, &rx);
  rd->rxOrigin = rx.o;
  rd->rxDirection = rx.d;
  // find ray after shifting one pixel in the y direction
  CameraSample sshift_y = sample;
  ++(sshift_y.pFilm.y);
  HRay ry;
  real_t wty = generateRay(sshift_y, &ry);
  rd->ryOrigin = ry.o;
  rd->ryDirection = ry.d;
  if (wtx == 0.f || wty == 0.f)
    return 0.f;
  rd->hasDifferentials = true;
  return wt;
}

ProjectiveCamera::ProjectiveCamera(const AnimatedTransform &cam2world,
                                   const HTransform &cam2scr,
                                   const bounds2 &screenWindow, real_t sopen,
                                   real_t sclose, real_t lensr, real_t focald,
                                   Film *f)
    : Camera(cam2world, sopen, sclose, f) {
  // initialize depth of field parameters
  lensRadius = lensr;
  focalDistance = focald;
  // compute projective camera transformations
  cameraToScreen = cam2scr;
  // compute projective camera screen transformations
  screenToRaster =
      ponos::scale(real_t(film->fullResolution.x),
                   real_t(film->fullResolution.y), 1.f) *
      ponos::scale(1.f / (screenWindow.upper.x - screenWindow.lower.x),
                   1.f / (screenWindow.lower.y - screenWindow.upper.y), 1.f) *
      ponos::translate(
          ponos::vec3(-screenWindow.lower.x, -screenWindow.upper.y, 0.f));
  rasterToScreen = inverse(screenToRaster);
  rasterToCamera = inverse(cameraToScreen) * rasterToScreen;
}

} // namespace helios
