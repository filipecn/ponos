#ifndef AERGIA_UI_TRACKBALL_INTERFACE_H
#define AERGIA_UI_TRACKBALL_INTERFACE_H

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

} // aergia namespace

#endif // AERGIA_UI_TRACKBALL_INTERFACE_H
