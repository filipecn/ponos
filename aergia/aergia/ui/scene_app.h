/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#ifndef AERGIA_UI_SCENE_APP_H
#define AERGIA_UI_SCENE_APP_H

#include <aergia/scene/scene.h>
#include <aergia/ui/app.h>

namespace aergia {

/** \brief Simple scene with viewports support.
 */
template <template <typename> class StructureType = ponos::Array>
class SceneApp : public App {
public:
  /** \brief Constructor.
   * \param w **[in]** window width (in pixels)
   * \param h **[in]** window height (in pixels)
   * \param t **[in]** window title
   * \param defaultViewport **[in | optional]** if true, creates a viewport with
   * the same size of the window
   */
  explicit SceneApp(uint w, uint h, const char *t, bool defaultViewport = true)
      : App(w, h, t, defaultViewport) {
    selectedObject = nullptr;
    activeObjectViewport = false;
  }
  virtual ~SceneApp() {}

  Scene<StructureType> scene; //!< object container

protected:
  void render() override {
    for (size_t i = 0; i < viewports.size(); i++) {
      viewports[i].render();
      scene.render();
    }
    if (this->renderCallback)
      this->renderCallback();
  }

  void mouse(double x, double y) override {
    App::mouse(x, y);
    if (selectedObject && selectedObject->active) {
      selectedObject->mouse(*viewports[activeObjectViewport].camera.get(),
                            viewports[activeObjectViewport].getMouseNPos());
      return;
    }
    activeObjectViewport = -1;
    for (size_t i = 0; i < viewports.size(); i++) {
      ponos::Point2 p = viewports[i].getMouseNPos();
      if (p >= ponos::Point2(-1.f, -1.f) && p <= ponos::Point2(1.f, 1.f)) {
        ponos::Ray3 r = viewports[i].camera->pickRay(p);
        if (selectedObject)
          selectedObject->selected = false;
        selectedObject = scene.intersect(r);
        if (selectedObject) {
          selectedObject->selected = true;
          activeObjectViewport = i;
          break;
        }
      }
    }
    // scene.transform = trackball.tb.transform * scene.transform;
  }

  void button(int b, int a) override {
    App::button(b, a);
    if (selectedObject)
      selectedObject->button(*viewports[activeObjectViewport].camera.get(),
                             viewports[activeObjectViewport].getMouseNPos(), b,
                             a);
  }

  int activeObjectViewport;
  SceneObject *selectedObject;
};

} // aergia namespace

#endif // AERGIA_UI_SCENE_APP_H
