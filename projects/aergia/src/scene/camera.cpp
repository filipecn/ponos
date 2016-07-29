#include <scene/camera.h>
#include <utils/open_gl.h>

#include <ponos.h>

using namespace ponos;

namespace aergia {

	Camera::Camera() {
		pos = Point3(0.f, 10.f, 1.5f);
		target = Point3(0.f, 0.f, 0.f);
		up = vec3(0.f, 1.f, 0.f);
		zoom = 1.f;
	}

	void Camera::look() {
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float pm[16];
		projection.matrix().column_major(pm);
		glMultMatrixf(pm);
		//gluPerspective(45.f, ratio, 0.1f, 1000.f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		float vm[16];
		view.matrix().column_major(vm);
		//gluLookAt(pos.x, pos.y, pos.z,
		//		target.x, target.y, target.z,
		//		0, 1, 0);
		glMultMatrixf(vm);
	}

	void Camera::resize(float w, float h) {
		display = vec2(w, h);
		ratio = w / h;
		clipSize = vec2(zoom, zoom);
		if(w < h)
			clipSize.y = clipSize.x / ratio;
		else clipSize.x = clipSize.y * ratio;
		projection = ponos::perspective(45.f, ratio, 0.1f, 1000.f);
		projection.computeInverse();
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

	void Camera::update() {
		view = ponos::lookAtRH(pos, target, up);
		view.computeInverse();
		model.computeInverse();
	}

	Transform Camera::getTransform() const {
		return model * view * projection;
	}

	ponos::Point3 Camera::viewPointOnWorldCoord() const {
		// TODO this product uses transpose... the result should be the same of target
		return ponos::inverse(model * view) * ponos::Point3(0.f, 0.f, 0.f);
	}

	ponos::Line Camera::viewLineFromWindow(ponos::Point2 p) const {
		return Line();
	}
} // aergia namespace
