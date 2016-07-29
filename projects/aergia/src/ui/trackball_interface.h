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
				modes.emplace_back(new PanMode());
				modes.emplace_back(new ZMode());
				curMode = 0;
			}

			virtual ~TrackballInterface() {}

			void draw();

			void buttonRelease(Camera& camera, int button);
			void buttonPress(const Camera& camera, int button);
			void mouseMove(Camera& camera);
			void mouseScroll(Camera& camera, ponos::vec2 d);

		protected:
			Trackball tb;

			size_t curMode;
			std::vector<TrackMode*> modes;
	};

	void TrackballInterface::buttonRelease(Camera& camera, int button) {

	}

	void TrackballInterface::buttonPress(const Camera& camera, int button) {
	}
	void TrackballInterface::mouseMove(Camera& camera) {

	}
} // aergia namespace

