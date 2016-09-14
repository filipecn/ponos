#include "helpers/geometry_drawers.h"

namespace aergia {

	void draw_segment(ponos::Segment3 segment) {
		glBegin(GL_LINES);
		glVertex(segment.a);
		glVertex(segment.b);
		glEnd();
	}

	void draw_circle(ponos::Circle circle) {
		glBegin(GL_TRIANGLE_FAN);
		glVertex(circle.c);
		float angle = 0.0;
		float step = PI_2 / 10.f;
		while(angle < PI_2 + step) {
			ponos::vec2 pp(circle.r * cosf(angle), circle.r * sinf(angle));
			glVertex(circle.c + pp);
			angle += step;
		}
		glEnd();
	}

	void draw_sphere(ponos::Sphere sphere) {
		const float vStep = PI / 20.f;
		const float hStep = PI_2 / 20.f;
		glBegin(GL_TRIANGLES);
		// south pole
		ponos::Point3 pole(0.f, -sphere.r, 0.f);
		for(float angle = 0.f; angle < PI_2; angle += hStep) {
			float r = sphere.r * sinf(vStep);
			glVertex(pole);
			glVertex(pole + r * ponos::vec3(cosf(angle), sinf(vStep), sinf(angle)));
			glVertex(pole + r * ponos::vec3(cosf(angle + hStep), sinf(vStep), sinf(angle + hStep)));
		}
		// north pole
		pole = ponos::Point3(0.f, sphere.r, 0.f);
		for(float angle = 0.f; angle < PI_2; angle += hStep) {
			float r = sphere.r * sinf(vStep);
			glVertex(pole);
			glVertex(pole + r * ponos::vec3(cosf(angle), -sinf(vStep), sinf(angle)));
			glVertex(pole + r * ponos::vec3(cosf(angle + hStep), -sinf(vStep), sinf(angle + hStep)));
		}

		glEnd();
		glBegin(GL_QUADS);
		for(float vAngle = vStep; vAngle <= PI - vStep; vAngle += vStep) {
			float r = sphere.r * sinf(vAngle);
			float R = sphere.r * sinf(vAngle + vStep);
			for(float angle = 0.f; angle < PI_2; angle += hStep) {
				glVertex(sphere.c + ponos::vec3(r * cosf(angle), sphere.r * cosf(vAngle), r * sinf(angle)));
				glVertex(sphere.c + ponos::vec3(r * cosf(angle + hStep), sphere.r * cosf(vAngle), r * sinf(angle + hStep)));
				glVertex(sphere.c + ponos::vec3(R * cosf(angle + hStep), sphere.r * cosf(vAngle + vStep), R * sinf(angle + hStep)));
				glVertex(sphere.c + ponos::vec3(R * cosf(angle), sphere.r * cosf(vAngle + vStep), R * sinf(angle)));
			}
		}
		glEnd();
	}

} // aergia namespace
