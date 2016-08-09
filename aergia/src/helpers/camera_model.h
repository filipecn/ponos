#pragma once

#include "scene/camera.h"
#include "utils/open_gl.h"

#include <ponos.h>

#include <iostream>

namespace aergia {

	class CameraModel {
  	public:
	 		CameraModel() {}
			virtual ~CameraModel() {}

			static void drawCamera(const Camera& camera) {
				ponos::vec3 dir = normalize(camera.target - camera.pos);
				ponos::vec3 left = normalize(cross(normalize(camera.up), dir));
				ponos::vec3 up = normalize(cross(dir, left));

				ponos::Point3 nbl, nbr, ntl, ntr;
				ponos::Point3 fbl, fbr, ftl, ftr;
				float talpha = tanf(camera.fov / 2.f);
				float tbeta = tanf((camera.fov / camera.ratio) / 2.f);
				nbl = camera.pos + dir * camera.near - camera.near * talpha * left - camera.near * tbeta * up;
				nbr = camera.pos + dir * camera.near + camera.near * talpha * left - camera.near * tbeta * up;
				ntl = camera.pos + dir * camera.near - camera.near * talpha * left + camera.near * tbeta * up;
				ntr = camera.pos + dir * camera.near + camera.near * talpha * left + camera.near * tbeta * up;

				fbl = camera.pos + dir * camera.far - camera.far * talpha * left - camera.far * tbeta * up;
				fbr = camera.pos + dir * camera.far + camera.far * talpha * left - camera.far * tbeta * up;
				ftl = camera.pos + dir * camera.far - camera.far * talpha * left + camera.far * tbeta * up;
				ftr = camera.pos + dir * camera.far + camera.far * talpha * left + camera.far * tbeta * up;
				static int i = 0;
				if(!i++) {
					std::cout << camera.pos;
					std::cout << nbl;
					std::cout << nbr;
					std::cout << ntl;
					std::cout << ntr;
				}
				glColor3f(0,0,0);
				glBegin(GL_LINES);
				glVertex(camera.pos); glVertex(fbl);
				glVertex(camera.pos); glVertex(fbr);
				glVertex(camera.pos); glVertex(ftl);
				glVertex(camera.pos); glVertex(ftr);
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

