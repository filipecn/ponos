#include <aergia/io/graphics_display.h>
#include <aergia/scene/camera_2d.h>
#include <aergia/ui/app.h>

namespace aergia {

App::App(uint w, uint h, const char *t, bool defaultViewport)
    : initialized(false), windowWidth(w), windowHeight(h), title(t) {
  if (defaultViewport)
    addViewport(0, 0, windowWidth, windowHeight);
  renderCallback = nullptr;
  mouseCallback = nullptr;
  buttonCallback = nullptr;
  keyCallback = nullptr;
}

size_t App::addViewport(uint x, uint y, uint w, uint h) {
  viewports.emplace_back(x, y, w, h);
  viewports[viewports.size() - 1].camera.reset(new Camera());
  static_cast<aergia::Camera *>(viewports[viewports.size() - 1].camera.get())
      ->resize(w, h);
  return viewports.size() - 1;
}

size_t App::addViewport2D(uint x, uint y, uint w, uint h) {
  viewports.emplace_back(x, y, w, h);
  viewports[viewports.size() - 1].camera.reset(new aergia::Camera2D());
  static_cast<aergia::Camera2D *>(viewports[viewports.size() - 1].camera.get())
      ->resize(w, h);
  return viewports.size() - 1;
}

void App::init() {
  if (initialized)
    return;
  GraphicsDisplay &gd =
      createGraphicsDisplay(windowWidth, windowHeight, title.c_str());
  gd.registerRenderFunc([this]() { render(); });
  gd.registerButtonFunc([this](int b, int a) { button(b, a); });
  gd.registerMouseFunc([this](double x, double y) { mouse(x, y); });
  gd.registerScrollFunc([this](double dx, double dy) { scroll(dx, dy); });
  gd.registerKeyFunc([this](int k, int a) { key(k, a); });
  initialized = true;
}

void App::run() {
  if (!initialized)
    init();
  GraphicsDisplay::instance().start();
}

void App::exit() { GraphicsDisplay::instance().stop(); }

void App::render() {
  for (size_t i = 0; i < viewports.size(); i++)
    viewports[i].render();
  if (renderCallback)
    renderCallback();
}

void App::button(int button, int action) {
  for (size_t i = 0; i < viewports.size(); i++)
    viewports[i].button(button, action);
  if (buttonCallback)
    buttonCallback(button, action);
}

void App::mouse(double x, double y) {
  for (size_t i = 0; i < viewports.size(); i++)
    viewports[i].mouse(x, y);
  if (mouseCallback)
    mouseCallback(x, y);
}

void App::scroll(double dx, double dy) {
  for (size_t i = 0; i < viewports.size(); i++)
    viewports[i].scroll(dx, dy);
  if (scrollCallback)
    scrollCallback(dx, dy);
}

void App::key(int k, int action) {
  for (size_t i = 0; i < viewports.size(); i++)
    viewports[i].key(k, action);
  if (keyCallback)
    keyCallback(k, action);
}

} // aergia namespace
