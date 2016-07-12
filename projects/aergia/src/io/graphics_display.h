#pragma once

#include <ponos.h>
#include <utils/open_gl.h>

#include <functional>
#include <memory>

namespace aergia {

  class GraphicsDisplay {
  public:
    ~GraphicsDisplay();
    static GraphicsDisplay& instance() {
      return instance_;
    }

    void set(int w, int h, const char* windowTitle);
    void getWindowSize(int &w, int &h);
    ponos::Point2 getMousePos();
    ponos::Point2 getMouseNPos();

    // run
    void start();
    void stop();
    // IO
    void registerRenderFunc(void (*f)());
    void registerButtonFunc(void (*f)(int,int));
    void registerKeyFunc(void (*f)(int,int));
    void registerMouseFunc(void (*f)(double,double));
    void registerScrollFunc(void (*f)(double,double));
    void registerResizeFunc(void (*f)(int,int));
    // graphics
    void clearScreen(float r, float g, float b, float a);
  private:
    static GraphicsDisplay instance_;
    GraphicsDisplay();
    GraphicsDisplay(GraphicsDisplay const&) = delete;
    void operator=(GraphicsDisplay const&) = delete;

    bool init();

    // window
    GLFWwindow* window;
    const char* title;
    int width, height;

    // USER CALLBACKS
    std::function<void()> renderCallback;
    std::function<void(int,int)> buttonCallback;
    std::function<void(int,int)> keyCallback;
    std::function<void(double,double)> mouseCallback;
    std::function<void(double,double)> scrollCallback;
    std::function<void(int,int)> resizeCallback;

    // DEFAULT CALLBACKS
    void buttonFunc(int button, int action);
    void keyFunc(int key, int action);
    void mouseFunc(double x, double y);
    void scrollFunc(double x, double y);
    void resizeFunc(int w, int h);

    // CALLBACKS
    static void error_callback(int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void button_callback(GLFWwindow* window, int button, int action, int mods);
    static void pos_callback(GLFWwindow* window, double x, double y);
    static void scroll_callback(GLFWwindow* window, double x, double y);
    static void resize_callback(GLFWwindow* window, int w, int h);
  };

  inline GraphicsDisplay& createGraphicsDisplay(int w, int h, const char* windowTitle) {
    GraphicsDisplay &gd = GraphicsDisplay::instance();
    gd.set(w, h, windowTitle);
    return gd;
  }

} // aergia namespace
