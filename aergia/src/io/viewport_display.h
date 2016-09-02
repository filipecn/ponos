#ifndef AERGIA_IO_VIEWPORT_DISPLAY_H
#define AERGIA_IO_VIEWPORT_DISPLAY_H

#include "scene/camera.h"

#include <ponos.h>

#include <functional>
#include <memory>

namespace aergia {

	class ViewportDisplay {
		public:
			ViewportDisplay(int _x, int _y, int _width, int _height);
			virtual ~ViewportDisplay() {}
			// display
			ponos::Point2 getMouseNPos();
			ponos::Point3 viewCoordToNormDevCoord(ponos::Point3 p);
			ponos::Point3 unProject(const Camera& c, ponos::Point3 p);
			void render();

			// USER CALLBACKS
			std::function<void()> renderCallback;

			// viewport
			int x, y, width, height;

			std::shared_ptr<Camera> camera;
	};

} // aergia namespace

#endif // AERGIA_IO_VIEWPORT_DISPLAY_H
