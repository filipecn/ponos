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
  resizeCallback = nullptr;
}

size_t App::addViewport(uint x, uint y, uint w, uint h) {
  viewports.emplace_back(x, y, w, h);
  viewports[viewports.size() - 1].camera.reset(new UserCamera3D());
  TrackballInterface::createDefault3D(viewports[viewports.size() - 1].camera->trackball);
  dynamic_cast<UserCamera3D *>(viewports[viewports.size() - 1].camera.get())
      ->resize(w, h);
  return viewports.size() - 1;
}

size_t App::addViewport2D(uint x, uint y, uint w, uint h) {
  viewports.emplace_back(x, y, w, h);
  viewports[viewports.size() - 1].camera.reset(new UserCamera2D());
  TrackballInterface::createDefault2D(viewports[viewports.size() - 1].camera->trackball);
  dynamic_cast<UserCamera2D *>(viewports[viewports.size() - 1].camera.get())
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
  gd.registerResizeFunc([this](int w, int h) { resize(w, h); });
  initialized = true;
}

void App::run() {
  if (!initialized)
    init();
  GraphicsDisplay::instance().start();
}

void App::exit() { GraphicsDisplay::instance().stop(); }

void App::render() {
  for (auto &viewport : viewports)
    viewport.render();
  if (renderCallback)
    renderCallback();
}

void App::button(int button, int action) {
  for (auto &viewport : viewports) {
    if (action == GLFW_PRESS && !viewport.hasMouseFocus())
      continue;
    viewport.button(button, action);
  }
  if (buttonCallback)
    buttonCallback(button, action);
}

void App::mouse(double x, double y) {
  for (auto &viewport : viewports)
    viewport.mouse(x, y);
  if (mouseCallback)
    mouseCallback(x, y);
}

void App::scroll(double dx, double dy) {
  for (auto &viewport : viewports)
    if (viewport.hasMouseFocus())
      viewport.scroll(dx, dy);
  if (scrollCallback)
    scrollCallback(dx, dy);
}

void App::key(int k, int action) {
  for (auto &viewport : viewports)
    viewport.key(k, action);
  if (keyCallback)
    keyCallback(k, action);
  else if (k == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    GraphicsDisplay::instance().stop();
}

void App::resize(int w, int h) {
  float wRatio = static_cast<float>(w) / windowWidth;
  float hRatio = static_cast<float>(h) / windowHeight;
  windowWidth = static_cast<uint>(w);
  windowHeight = static_cast<uint>(h);
  for (auto &viewport : viewports) {
    viewport.x *= wRatio;
    viewport.y *= hRatio;
    viewport.width *= wRatio;
    viewport.height *= hRatio;
    if (viewport.camera)
      viewport.camera->resize(viewport.width, viewport.height);
  }
}

} // namespace aergia
