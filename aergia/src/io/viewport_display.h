#ifndef AERGIA_IO_VIEWPORT_DISPLAY_H
#define AERGIA_IO_VIEWPORT_DISPLAY_H

#include "scene/camera.h"

#include <ponos.h>

#include <functional>
#include <memory>

namespace aergia {

	/* display region
	 * Defines a region in the screen.
	 */
	class ViewportDisplay {
		public:
			/* Constructor
			 * @_x **[in]** start pixel in X
			 * @_y **[in]** start pixel in Y
			 * @_width **[in]** width (in pixels)
			 * @_height **[in]** height (in pixels)
			 */
			ViewportDisplay(int _x, int _y, int _width, int _height);
			virtual ~ViewportDisplay() {}
			/* get
			 * The point **(x, y)** is mapped to **(0, 0)** and **(x + width, y + height)** to **(1, 1)**
			 * @return normalized mouse coordinates relative to the viewport
			 */
			ponos::Point2 getMouseNPos();
			/* convert
			 * @p **[in]** point (in view space)
			 * @return **p** mapped to NDC (**[-1,1]**)
			 */
			ponos::Point3 viewCoordToNormDevCoord(ponos::Point3 p);
			/* convert
			 * @c **[in]** camera
			 * @p **[in]** point (in screen space)
			 * @return the unprojected point by the inverse of the camera transform to world space
			 */
			ponos::Point3 unProject(const Camera& c, ponos::Point3 p);

			void render();

			// render callback
			std::function<void()> renderCallback;

			// viewport
			int x, y, width, height;

			std::shared_ptr<Camera> camera;
	};

} // aergia namespace

#endif // AERGIA_IO_VIEWPORT_DISPLAY_H
