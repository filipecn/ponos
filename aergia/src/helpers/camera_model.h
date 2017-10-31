#ifndef AERGIA_HELPERS_CAMERA_MODEL_H
#define AERGIA_HELPERS_CAMERA_MODEL_H

#include "scene/camera.h"
#include "utils/open_gl.h"

#include <ponos.h>

#include <iostream>

namespace aergia {

class CameraModel {
public:
  CameraModel() {}
  virtual ~CameraModel() {}

  static void drawCamera(const Camera &camera) {
    ponos::vec3 dir = normalize(camera.target - camera.pos);
    ponos::vec3 left = normalize(cross(normalize(camera.up), dir));
    ponos::vec3 up = normalize(cross(dir, left));

    ponos::Point3 nbl, nbr, ntl, ntr;
    ponos::Point3 fbl, fbr, ftl, ftr;
    float talpha = tanf(
        static_cast<PerspectiveProjection *>(camera.projection.get())->fov /
        2.f);
    float tbeta = tanf(
        (static_cast<PerspectiveProjection *>(camera.projection.get())->fov /
         camera.projection->ratio) /
        2.f);
    nbl = camera.pos + dir * camera.getNear() -
          camera.getNear() * talpha * left - camera.getNear() * tbeta * up;
    nbr = camera.pos + dir * camera.getNear() +
          camera.getNear() * talpha * left - camera.getNear() * tbeta * up;
    ntl = camera.pos + dir * camera.getNear() -
          camera.getNear() * talpha * left + camera.getNear() * tbeta * up;
    ntr = camera.pos + dir * camera.getNear() +
          camera.getNear() * talpha * left + camera.getNear() * tbeta * up;

    fbl = camera.pos + dir * camera.getFar() - camera.getFar() * talpha * left -
          camera.getFar() * tbeta * up;
    fbr = camera.pos + dir * camera.getFar() + camera.getFar() * talpha * left -
          camera.getFar() * tbeta * up;
    ftl = camera.pos + dir * camera.getFar() - camera.getFar() * talpha * left +
          camera.getFar() * tbeta * up;
    ftr = camera.pos + dir * camera.getFar() + camera.getFar() * talpha * left +
          camera.getFar() * tbeta * up;
    static int i = 0;
    if (!i++) {
      std::cout << camera.pos;
      std::cout << nbl;
      std::cout << nbr;
      std::cout << ntl;
      std::cout << ntr;
    }
    glColor3f(0, 0, 0);
    glBegin(GL_LINES);
    glVertex(camera.pos);
    glVertex(fbl);
    glVertex(camera.pos);
    glVertex(fbr);
    glVertex(camera.pos);
    glVertex(ftl);
    glVertex(camera.pos);
    glVertex(ftr);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex(nbl);
    glVertex(nbr);
    glVertex(ntr);
    glVertex(ntl);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glBegin(GL_LINE_LOOP);
    glVertex(fbl);
    glVertex(fbr);
    glVertex(ftr);
    glVertex(ftl);
    glEnd();
  }
};

} // aergia namespace

#endif // AERGIA_HELPERS_CAMERA_MODEL_H
