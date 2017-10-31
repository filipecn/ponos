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

#ifndef AERGIA_IO_GRAPHICS_DISPLAY_H
#define AERGIA_IO_GRAPHICS_DISPLAY_H

#include "scene/camera.h"
#include "utils/open_gl.h"

#include <ponos.h>

#include <functional>
#include <memory>

namespace aergia {

/** singleton
 *
 * Main class for screen operations. Responsable for getting user input and
 * rendering. The screen origin **(0, 0)** is the lower-left corner of the
 * window.
 */
class GraphicsDisplay {
public:
  ~GraphicsDisplay();
  static GraphicsDisplay &instance() { return instance_; }
  /* set
   * \param w **[in]** width (in pixels)
   * \param h **[in]** height (in pixels)
   * \param windowTitle **[in]**
   * Set window properties.
   */
  void set(int w, int h, const char *windowTitle);
  /* get
   * \param w **[out]** receives width value
   * \param h **[out]** receives height value
   */
  void getWindowSize(int &w, int &h);
  /* get
   * \returns mouse position (screen space)
   */
  ponos::Point2 getMousePos();
  /* get
   * \returns mouse position (NDC **[-1, 1]**)
   */
  ponos::Point2 getMouseNPos();
  /** \brief convert
   * \param p **[in]** point (in norm dev coordinates)
   * \returns **p** mapped to view coordinates
   */
  ponos::Point3 normDevCoordToViewCoord(ponos::Point3 p);
  /* convert
   * \param p **[in]** point (in view space)
   * \returns **p** mapped to NDC (**[-1,1]**)
   */
  ponos::Point3 viewCoordToNormDevCoord(ponos::Point3 p);
  /* convert
   * \param c **[in]** camera
   * \param p **[in]** point (in screen space)
   * \returns the unprojected point by the inverse of the camera transform to
   * world space
   */
  ponos::Point3 unProject(const Camera &c, ponos::Point3 p);
  /* main loop
   * Starts the application, opens the window and enters in the main loop.
   */
  void start();
  /* exit
   * Closes the window.
   */
  void stop();
  /* query
   * \returns **true** if application is running
   */
  bool isRunning();
  // IO
  void registerRenderFunc(void (*f)());
  void registerRenderFunc(std::function<void()> f);
  void registerButtonFunc(void (*f)(int, int));
  void registerButtonFunc(std::function<void(int, int)> f);
  void registerKeyFunc(void (*f)(int, int));
  void registerKeyFunc(std::function<void(int, int)> f);
  void registerMouseFunc(void (*f)(double, double));
  void registerMouseFunc(std::function<void(double, double)> f);
  void registerScrollFunc(void (*f)(double, double));
  void registerScrollFunc(std::function<void(double, double)> f);
  void registerResizeFunc(void (*f)(int, int));
  // graphics
  void beginFrame();
  void endFrame();
  /* clear screen
   * \param r **[in]** red
   * \param g **[in]** green
   * \param b **[in]** blue
   * \param a **[in]** alpha
   * Assign the given color **(r, g, b, a)** to all pixels of the screen
   */
  void clearScreen(float r, float g, float b, float a);
  // events
  void processInput();
  // user input
  int keyState(int key);

private:
  static GraphicsDisplay instance_;
  GraphicsDisplay();
  GraphicsDisplay(GraphicsDisplay const &) = delete;
  void operator=(GraphicsDisplay const &) = delete;

  bool init();

  // window
  GLFWwindow *window;
  const char *title;
  int width, height;

  // USER CALLBACKS
  std::function<void()> renderCallback;
  std::function<void(int, int)> buttonCallback;
  std::function<void(int, int)> keyCallback;
  std::function<void(double, double)> mouseCallback;
  std::function<void(double, double)> scrollCallback;
  std::function<void(int, int)> resizeCallback;

  // DEFAULT CALLBACKS
  void buttonFunc(int button, int action);
  void keyFunc(int key, int action);
  void mouseFunc(double x, double y);
  void scrollFunc(double x, double y);
  void resizeFunc(int w, int h);

  // CALLBACKS
  static void error_callback(int error, const char *description);
  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods);
  static void button_callback(GLFWwindow *window, int button, int action,
                              int mods);
  static void pos_callback(GLFWwindow *window, double x, double y);
  static void scroll_callback(GLFWwindow *window, double x, double y);
  static void resize_callback(GLFWwindow *window, int w, int h);
};

/* create
 * \param w **[in]** window's width (in pixels)
 * \param h **[in]** window's height (in pixels)
 * \param windowTitle **[in]** window's title
 * \returns reference to GraphicsDiplay with the window created
 */
inline GraphicsDisplay &createGraphicsDisplay(int w, int h,
                                              const char *windowTitle) {
  GraphicsDisplay &gd = GraphicsDisplay::instance();
  gd.set(w, h, windowTitle);
  return gd;
}

} // aergia namespace

#endif // AERGIA_IO_GRAPHICS_DISPLAY_H
