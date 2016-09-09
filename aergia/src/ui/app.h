#ifndef AERGIA_UI_APP_H
#define AERGIA_UI_APP_H

#include "io/graphics_display.h"
#include "io/viewport_display.h"

#include <string>
#include <vector>

namespace aergia {

	/* base class
	 * An App makes the creation of viewports easy.
	 */
	class App {
		public:
			/* Constructor.
			 * @w **[in]** window width (in pixels)
			 * @h **[in]** window height (in pixels)
			 * @t **[in]** window title
			 * @defaultViewport **[in | optional]** if true, creates a viewport with the same size of the window
			 */
			explicit App(uint w, uint h, const char* t, bool defaultViewport = true);
			virtual ~App() {}
			/* add
			 * @x **[in]** first pixel in X
			 * @y **[in]** first pixel in Y
			 * @w **[in]** viewport width (in pixels)
			 * @h **[in]** viewport height (in pixels)
			 * Creates a new viewport **[x, y, w, h]**.
			 *
			 * **Note:** the origin of the screen space **(0, 0)** is on the upper-left corner of the window.
			 * @return the id of the new viewport
			 */
			size_t addViewport(uint x, uint y, uint w, uint h);
			void init();
			void run();

			std::vector<ViewportDisplay> viewports;

		protected:
			bool initialized;
			uint windowWidth, windowHeight;
			std::string title;

			void processInput();
			void render();
	};

} // aergia namespace

#endif // AERGIA_UI_APP_H
