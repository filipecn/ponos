#include "io/graphics_display.h"

namespace aergia {

  GraphicsDisplay GraphicsDisplay::instance_;

  GraphicsDisplay::GraphicsDisplay()
  : window(nullptr),
  title(nullptr),
  width(400),
  height(400){
    renderCallback = nullptr;
    buttonCallback = nullptr;
    keyCallback = nullptr;
    mouseCallback = nullptr;
    scrollCallback = nullptr;
    keyCallback = nullptr;
  }

  GraphicsDisplay::~GraphicsDisplay(){
    glfwDestroyWindow(window);
    glfwTerminate();
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
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback (window, button_callback);
    glfwSetCursorPosCallback (window, pos_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowSizeCallback(window, resize_callback);
    return true;
  }

  void GraphicsDisplay::start(){
    while(!glfwWindowShouldClose(this->window)){

      glfwGetFramebufferSize(window, &this->width, &this->height);
      glViewport(0, 0, this->width, this->height);

      if(this->renderCallback) {
        this->renderCallback();
      }
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }

  void GraphicsDisplay::getWindowSize(int &w, int &h){
    w = this->width;
    h = this->height;
  }

  vec2 GraphicsDisplay::getMousePos(){
    double x, y;
    glfwGetCursorPos(this->window, &x, &y);
    return vec2(x, this->height - y);
  }

  void GraphicsDisplay::stop(){
  	glfwSetWindowShouldClose(window, GL_TRUE);
  }

  void GraphicsDisplay::clearScreen(float r, float g, float b, float a) {
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

  void GraphicsDisplay::error_callback(int error, const char* description){
  	fputs(description, stderr);
  }

  void GraphicsDisplay::registerRenderFunc(void (*f)()){
  	this->renderCallback = f;
  }

  /////////////////////////// KEY FUNCTIONS ////////////////////////////////////////////////////
  void GraphicsDisplay::registerKeyFunc(void (*f)(int,int)){
    this->keyCallback = f;
  }

  void GraphicsDisplay::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(instance_.keyCallback)
    instance_.keyCallback(key,action);
    else instance_.keyFunc(key,action);
  }

  void GraphicsDisplay::keyFunc(int key, int action){
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////// BUTTON FUNCTIONS //////////////////////////////////////////////////
  void GraphicsDisplay::registerButtonFunc(void (*f)(int,int)){
    this->buttonCallback = f;
  }

  void GraphicsDisplay::button_callback(GLFWwindow* window, int button, int action, int mods){
    if(instance_.buttonCallback)
    instance_.buttonCallback(button,action);
    else instance_.buttonFunc(button,action);
  }

  void GraphicsDisplay::buttonFunc(int button, int action){
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////// MOUSE MOTION FUNCTIONS ////////////////////////////////////////////
  void GraphicsDisplay::registerMouseFunc(void (*f)(double,double)){
    this->mouseCallback = f;
  }

  void GraphicsDisplay::pos_callback(GLFWwindow* window, double x, double y){
    if(instance_.mouseCallback)
    instance_.mouseCallback(x,y);
    else instance_.mouseFunc(x,y);
  }

  void GraphicsDisplay::mouseFunc(double x, double y){
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////// MOUSE SCROLL FUNCTIONS ////////////////////////////////////////////
  void GraphicsDisplay::registerScrollFunc(void (*f)(double,double)){
    this->scrollCallback = f;
  }

  void GraphicsDisplay::scroll_callback(GLFWwindow* window, double x, double y){
    if(instance_.scrollCallback)
    instance_.scrollCallback(x,y);
    else
    instance_.scrollFunc(x,y);
  }

  void GraphicsDisplay::scrollFunc(double x, double y){
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////
  void GraphicsDisplay::registerResizeFunc(void (*f)(int,int)){
    this->resizeCallback = f;
  }

  void GraphicsDisplay::resize_callback(GLFWwindow* window, int w, int h){
    instance_.resizeFunc(w,h);
    if(instance_.resizeCallback){
      instance_.getWindowSize(w,h);
      instance_.resizeCallback(w,h);
    }
  }

  void GraphicsDisplay::resizeFunc(int w, int h){
    glfwGetFramebufferSize(window, &this->width, &this->height);
  }

} // aergia namespace
