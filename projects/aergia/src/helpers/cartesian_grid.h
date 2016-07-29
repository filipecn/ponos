#pragma once

#include <ponos.h>
#include "utils/open_gl.h"

namespace aergia {

	class CartesianGrid {
		public:
			CartesianGrid() {}

			void setDimension(size_t d, int a, int b) {
				planes[d].low = a;
				planes[d].high = b;
			}

			ponos::Transform transform;
			ponos::Interval<int>  planes[3];

			void draw();
	};

	void CartesianGrid::draw() {
		glColor4f(0,0,0,0.5);
		glBegin(GL_LINES);
		// XY
		for(int x = planes[0].low; x <= planes[0].high; x++) {
			glVertex(transform(ponos::Point3(1.f * x, 1.f * planes[1].low,  0.f)));
			glVertex(transform(ponos::Point3(1.f * x, 1.f * planes[1].high, 0.f)));
		}
		for(int y = planes[1].low; y <= planes[1].high; y++) {
			glVertex(transform(ponos::Point3(1.f * planes[0].low,  1.f * y, 0.f)));
			glVertex(transform(ponos::Point3(1.f * planes[0].high, 1.f * y, 0.f)));
		}
		// YZ
		for(int y = planes[1].low; y <= planes[1].high; y++) {
			glVertex(transform(ponos::Point3(0.f, 1.f * y, 1.f * planes[2].low)));
			glVertex(transform(ponos::Point3(0.f, 1.f * y, 1.f * planes[2].high)));
		}
		for(int z = planes[2].low; z <= planes[2].high; z++) {
			glVertex(transform(ponos::Point3(0.f,  1.f * planes[1].low,  1.f * z)));
			glVertex(transform(ponos::Point3(0.f,  1.f * planes[1].high, 1.f * z)));
		}
		// XZ
		for(int x = planes[0].low; x <= planes[0].high; x++) {
			glVertex(transform(ponos::Point3(1.f * x, 0.f, 1.f * planes[2].low )));
			glVertex(transform(ponos::Point3(1.f * x, 0.f, 1.f * planes[2].high)));
		}
		for(int z = planes[2].low; z <= planes[2].high; z++) {
			glVertex(transform(ponos::Point3(1.f * planes[1].low,  0.f,  1.f * z)));
			glVertex(transform(ponos::Point3(1.f * planes[1].high, 0.f,  1.f * z)));
		}
		// axis
		glColor4f(1,0,0,1);
		glVertex(transform(ponos::Point3()));glVertex(transform(ponos::Point3(0.5,0,0)));
		glColor4f(0,1,0,1);
		glVertex(transform(ponos::Point3()));glVertex(transform(ponos::Point3(0,0.5,0)));
		glColor4f(0,0,1,1);
		glVertex(transform(ponos::Point3()));glVertex(transform(ponos::Point3(0,0,0.5)));
		glEnd();
	}

} // aergia namespace

