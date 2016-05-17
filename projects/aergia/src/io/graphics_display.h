#pragma once

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>

namespace aergia {

  class GraphicsDisplay {
  public:
    ~GraphicsDisplay() {}
    static GraphicsDisplay& instance() {
      static GraphicsDisplay instance_;
      return instance_;
    }

    void set(int w, int h, const char* windowTitle);

    // run
    void start();

  private:
    GraphicsDisplay();

    bool init();

    // window
    int width, height;
    GLFWwindow* window;
    const char* title;
  };

  inline GraphicsDisplay& createGraphicsDisplay(int w, int h, const char* windowTitle) {
    GraphicsDisplay &gd = GraphicsDisplay::instance();
    gd.set(w, h, windowTitle);
    return gd;
  }

} // aergia namespace
