#ifndef AERGIA_UI_APP_H
#define AERGIA_UI_APP_H

#include "io/graphics_display.h"
#include "io/viewport_display.h"

#include <string>
#include <vector>

namespace aergia {

/** \brief base class
 * An App makes the creation of viewports easy.
 */
class App {
public:
  /* Constructor.
   * \param w **[in]** window width (in pixels)
   * \param h **[in]** window height (in pixels)
   * \param t **[in]** window title
   * \param defaultViewport **[in | optional]** if true, creates a viewport with
   * the
   * same size of the window
   */
  explicit App(uint w, uint h, const char *t, bool defaultViewport = true);
  virtual ~App() {}
  /** \brief add
   * \param x **[in]** first pixel in X
   * \param y **[in]** first pixel in Y
   * \param w **[in]** viewport width (in pixels)
   * \param h **[in]** viewport height (in pixels)
   * Creates a new viewport **[x, y, w, h]**.
   *
   * **Note:** the origin of the screen space **(0, 0)** is on the upper-left
   *corner of the window.
   * \return the id of the new viewport
   */
  size_t addViewport(uint x, uint y, uint w, uint h);
  size_t addViewport2D(uint x, uint y, uint w, uint h);
  void init();
  void run();
  template <typename T> T *getCamera(size_t i = 0) {
    return static_cast<T *>(viewports[i].camera.get());
  }

  std::vector<ViewportDisplay> viewports;
  std::function<void()> renderCallback;
  std::function<void(double, double)> scrollCallback;
	std::function<void(double, double)> mouseCallback;
	std::function<void(int, int)> buttonCallback;

protected:
  bool initialized;
  uint windowWidth, windowHeight;
  std::string title;

  virtual void render();
  virtual void button(int b, int a);
  virtual void mouse(double x, double y);
  virtual void scroll(double dx, double dy);
};

} // aergia namespace

#endif // AERGIA_UI_APP_H
