#include "io/graphics_display.h"

namespace aergia {

  GraphicsDisplay::GraphicsDisplay()
    : window(nullptr),
      title(nullptr),
      width(400),
      height(400){
  }

  void GraphicsDisplay::set(int w, int h, const char* windowTitle) {
    width = w;
    height = h;
    title = windowTitle;
    window = nullptr;

    init();
  }

  bool GraphicsDisplay::init() {
    if (!glfwInit())
      return false;
    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window){
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(window);
    return true;
  }

  void GraphicsDisplay::start() {
    while(!glfwWindowShouldClose(window)){
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }

} // aergia namespace
