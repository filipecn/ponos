#ifndef AERGIA_UI_SCENE_APP_H
#define AERGIA_UI_SCENE_APP_H

#include "scene/scene.h"
#include "ui/app.h"

namespace aergia {

	/* derived class
	 * Simple scene with viewports support.
	 */
	template<template <typename> class StructureType = ponos::Array>
		class SceneApp : public App {
			public:
				/* Constructor.
				 * @w **[in]** window width (in pixels)
				 * @h **[in]** window height (in pixels)
				 * @t **[in]** window title
				 * @defaultViewport **[in | optional]** if true, creates a viewport with the same size of the window
				 */
				explicit SceneApp(uint w, uint h, const char* t, bool defaultViewport = true)
					: App(w, h, t, defaultViewport) {
						selectedObject = nullptr;
					}
				virtual ~SceneApp() {}

				Scene<StructureType> scene;

			protected:
				void render() override {
					App::render();
					scene.render();
				}

				void mouse(double x, double y) override {
					App::mouse(x, y);
					ponos::Ray3 r = viewports[0].camera->pickRay(
							viewports[0].getMouseNPos());
					if(selectedObject)
						selectedObject->selected = false;
					selectedObject = scene.intersect(r);
					if(selectedObject)
						selectedObject->selected = true;
					scene.transform = trackball.tb.transform * scene.transform;
				}

				SceneObject *selectedObject;
		};

} // aergia namespace

#endif // AERGIA_UI_SCENE_APP_H

