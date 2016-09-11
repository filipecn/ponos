#include "ui/trackball_interface.h"

namespace aergia {

	void TrackballInterface::draw() {
		modes[curMode]->draw(tb);
	}

	void TrackballInterface::buttonRelease(Camera& camera, int button, ponos::Point2 p) {
		modes[curMode]->stop(tb, camera, p);
	}

	void TrackballInterface::buttonPress(const Camera& camera, int button, ponos::Point2 p) {
		modes[curMode]->start(tb, camera, p);
	}

	void TrackballInterface::mouseMove(Camera& camera, ponos::Point2 p) {
		modes[curMode]->update(tb, camera, p);
	}

} // aergia namespace
