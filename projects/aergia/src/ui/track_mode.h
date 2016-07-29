#pragma once

#include "scene/camera.h"
#include "ui/trackball.h"

#include <ponos.h>

namespace aergia {

	class TrackMode {
		public:
			virtual ~TrackMode() {}

			virtual void draw(Trackball tb) {}

			// CONTROL
			virtual void start(Trackball &tb, const Camera& camera) {}
			virtual void update(Trackball &tb, Camera& camera) = 0;
			virtual void stop(Trackball &tb, Camera& camera) {}


		protected:
			ponos::Point3 start_;
			ponos::Point3 end_;
	};

	class PanMode : public TrackMode {
		public:
			~PanMode(){}

			void draw(Trackball tb) {}

			void update(Trackball &tb, Camera& camera) override {}
	};

	class ZMode : public TrackMode {
		public:
			~ZMode() {}

			void update(Trackball &tb, Camera& camera) override {

			}
	};

} // aergia namespace

