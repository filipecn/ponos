#include "ui/app.h"

#include "io/graphics_display.h"

namespace aergia {

	App::App(uint w, uint h, const char* t, bool defaultViewport)
		: initialized(false), windowWidth(w), windowHeight(h), title(t) {
			if(defaultViewport)
				addViewport(0, 0, windowWidth, windowHeight);
		}

	size_t App::addViewport(uint x, uint y, uint w, uint h) {
		viewports.emplace_back(x, y, w, h);
		return viewports.size() - 1;
	}

	void App::init() {
		if(initialized)
			return;
		GraphicsDisplay &gd = createGraphicsDisplay(windowWidth, windowHeight, title.c_str());
		gd.registerRenderFunc([this](){ render(); });
		viewports[0].camera->resize(windowWidth, windowHeight);
		initialized = true;
	}

	void App::run() {
		if(!initialized)
			init();
		GraphicsDisplay::instance().start();
	}

	void App::render() {
		for(size_t i = 0; i < viewports.size(); i++)
			viewports[i].render();
	}

	void App::processInput() {}


} // aergia namespace
