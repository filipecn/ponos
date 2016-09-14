#include "ui/app.h"

#include "io/graphics_display.h"

namespace aergia {

	App::App(uint w, uint h, const char* t, bool defaultViewport)
		: initialized(false), windowWidth(w), windowHeight(h), title(t) {
			if(defaultViewport) {
				addViewport(0, 0, windowWidth, windowHeight);
				viewports[0].camera.reset(new aergia::Camera());
			}
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
		gd.registerButtonFunc([this](int b, int a){ button(b, a); });
		gd.registerMouseFunc([this](double x, double y){ mouse(x, y); });
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

	void App::button(int button, int action) {
		if(action == GLFW_RELEASE)
			trackball.buttonRelease(*viewports[0].camera.get(), button, viewports[0].getMouseNPos());
		else
			trackball.buttonPress(*viewports[0].camera.get(), button, viewports[0].getMouseNPos());
	}

	void App::mouse(double x, double y) {
		trackball.mouseMove(*viewports[0].camera.get(), viewports[0].getMouseNPos());
		//viewports[0].camera->setPos(trackball.tb.transform(viewports[0].camera->getPos()));
		//viewports[0].camera->setTarget(trackball.tb.transform(viewports[0].camera->getTarget()));
		//trackball.tb.center = viewports[0].camera->getTarget();
		for(size_t i = 0; i < viewports.size(); i++)
			viewports[i].mouse(x, y);
	}

} // aergia namespace
