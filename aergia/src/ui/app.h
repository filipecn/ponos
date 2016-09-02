#ifndef AERGIA_UI_APP_H
#define AERGIA_UI_APP_H

#include "io/graphics_display.h"
#include "io/viewport_display.h"
#include "scene/scene_object.h"

namespace aergia {

	class App {
  	public:
	 		explicit App(size_t w, size_t h);
			virtual ~App() {}

			void processInput();
			void render();
	};

} // aergia namespace

#endif // AERGIA_UI_APP_H
