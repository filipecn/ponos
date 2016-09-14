#include "ui/trackball_interface.h"

namespace aergia {

	void TrackballInterface::draw() {
		modes[curMode]->draw(tb);
	}

	void TrackballInterface::buttonRelease(Camera& camera, int button, ponos::Point2 p) {
		modes[curMode]->stop(tb, camera, p);
	}

	void TrackballInterface::buttonPress(const Camera& camera, int button, ponos::Point2 p) {
		switch(button) {
			case GLFW_MOUSE_BUTTON_LEFT: curMode = 0; break;
			case GLFW_MOUSE_BUTTON_MIDDLE: curMode = 1; break;
			case GLFW_MOUSE_BUTTON_RIGHT: curMode = 2; break;
		}
		modes[curMode]->start(tb, camera, p);
	}

	void TrackballInterface::mouseMove(Camera& camera, ponos::Point2 p) {
		modes[curMode]->update(tb, camera, p);
	}

	void TrackballInterface::mouseScroll(Camera& camera, ponos::vec2 d) {
		//modes[curMode]->update(tb, camera, d);
	}

} // aergia namespace
