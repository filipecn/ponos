#pragma once

#include "scene/camera.h"
#include "ui/trackball.h"
#include "ui/track_mode.h"

#include <ponos.h>
#include <vector>

namespace aergia {

	class TrackballInterface {
		public:
			TrackballInterface() {
				modes.emplace_back(new RotateMode());
				modes.emplace_back(new ZMode());
				modes.emplace_back(new PanMode());
				curMode = 0;
			}

			virtual ~TrackballInterface() {}

			void draw();

			void buttonRelease(Camera& camera, int button, ponos::Point2 p);
			void buttonPress(const Camera& camera, int button, ponos::Point2 p);
			void mouseMove(Camera& camera, ponos::Point2 p);
			void mouseScroll(Camera& camera, ponos::vec2 d);

			Trackball tb;

		protected:
			size_t curMode;
			std::vector<TrackMode*> modes;
	};

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

