#include "io/graphics_display.h"
#include <scene/camera.h>
#include <utils/open_gl.h>

#include <ponos.h>

using namespace ponos;

namespace aergia {

	Camera::Camera() {
		pos = Point3(20.f, 0.f, 0.f);
		target = Point3(0.f, 0.f, 0.f);
		up = vec3(0.f, 1.f, 0.f);
		zoom = 1.f;
		near = 0.1f;
		far = 1000.f;
		fov = 45.f;
	}

	void Camera::look() {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float pm[16];
		projection.matrix().column_major(pm);
		glMultMatrixf(pm);
		//gluPerspective(45.f, ratio, 0.1f, 1000.f);
		GLdouble projMatrix[16];
		glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		float vm[16];
		view.matrix().column_major(vm);
		// gluLookAt(pos.x, pos.y, pos.z,
		// 		target.x, target.y, target.z,
		//		0, 1, 0);
		glMultMatrixf(vm);
		model.matrix().column_major(vm);
		glMultMatrixf(vm);
	}

	void Camera::resize(float w, float h) {
		display = vec2(w, h);
		ratio = w / h;
		clipSize = vec2(zoom, zoom);
		if(w < h)
			clipSize.y = clipSize.x / ratio;
		else clipSize.x = clipSize.y * ratio;
		projection = ponos::perspective(fov, ratio, near, far);
		update();
	}

	void Camera::setZoom(float z) {
		zoom = z;
		update();
	}

	void Camera::setPos(Point3 p) {
		pos = p;
		update();
	}

	void Camera::setTarget(Point3 p) {
		target = p;
		update();
	}

	void Camera::setFov(float f) {
		fov = f;
		projection = ponos::perspective(fov, ratio, near, far);
		update();
	}

	void Camera::setFar(float f) {
		far = f;
		projection = ponos::perspective(fov, ratio, near, far);
		update();
	}

	void Camera::setNear(float n) {
		near = n;
		projection = ponos::perspective(fov, ratio, near, far);
		update();
	}

	void Camera::update() {
		view = ponos::lookAtRH(pos, target, up);
		view.computeInverse();
		model.computeInverse();
		// update frustum
		frustum.set(model * view * projection);
	}

	Transform Camera::getTransform() const {
		return model * view * projection;
	}

	ponos::Line Camera::viewLineFromWindow(ponos::Point2 p) const {
		// TODO it would be more intuitively calculated by using the inverse of MVP
		/*ponos::vec3 dir = normalize(target - pos);
		ponos::vec3 left = normalize(cross(normalize(up), dir));
		ponos::vec3 new_up = normalize(cross(dir, left));
		float ta = near * tanf(fov / 2.f);
		float tb = near * tanf((fov / ratio) / 2.f);
		ponos::Point3 P = pos + dir * near - left * p.x * ta + new_up * p.y * tb;*/
		ponos::Point3 P = ponos::inverse(model * view)(ponos::inverse(projection) * ponos::Point3(p.x, p.y, -1.f));
		std::cout << "view line from plane\n";
		std::cout << P;
		std::cout << pos;
		return Line(pos, P - pos);
	}

	ponos::Plane Camera::viewPlane(ponos::Point3 p) const {
		ponos::vec3 n = pos - p;
		if(fabs(n.length()) < 1e-8)
			n = ponos::vec3(0,0,0);
		else n = ponos::normalize(n);
		return Plane(ponos::Normal(n), ponos::dot(n, ponos::vec3(p.x, p.y, p.z)));
	}

} // aergia namespace
