// Created by filipecn on 6/12/18.
#include <glad/glad.h>
//#include <Eigen/Core>
#include <nanogui/nanogui.h>
#define NANOGUI_GLAD 1

#include <aergia/aergia.h>
// GLFW
//
#if defined(NANOGUI_GLAD)
#if defined(NANOGUI_SHARED) && !defined(GLAD_GLAPI_EXPORT)
#define GLAD_GLAPI_EXPORT
#endif

#else
#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#else
#define GL_GLEXT_PROTOTYPES
#endif
#endif

using namespace nanogui;

enum test_enum {
  Item1 = 0,
  Item2,
  Item3
};

bool bvar = true;
int ivar = 12345678;
double dvar = 3.1415926;
float fvar = (float) dvar;
std::string strval = "A string";
test_enum enumval = Item2;
Color colval(0.5f, 0.5f, 0.7f, 1.f);

Screen *screen = nullptr;

int main(int /* argc */, char ** /* argv */) {

  aergia::SceneApp<> app(800, 800);
  app.init();
  auto window = aergia::GraphicsDisplay::instance().getGLFWwindow();
#if defined(NANOGUI_GLAD)
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    throw std::runtime_error("Could not initialize GLAD!");
  glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
#endif

  // Create a nanogui screen and pass the glfw pointer to initialize
  screen = new Screen();
  screen->initialize(window, true);

  // Create nanogui gui
  bool enabled = true;
  FormHelper *gui = new FormHelper(screen);
  ref<Window> nanoguiWindow = gui->addWindow(Eigen::Vector2i(10, 10), "Form helper example");
  gui->addGroup("Basic types");
  gui->addVariable("bool", bvar)->setTooltip("Test tooltip.");
  gui->addVariable("string", strval);

  gui->addGroup("Validating fields");
  gui->addVariable("int", ivar)->setSpinnable(true);
  gui->addVariable("float", fvar)->setTooltip("Test.");
  gui->addVariable("double", dvar)->setSpinnable(true);

  gui->addGroup("Complex types");
  gui->addVariable("Enumeration", enumval, enabled)->setItems({"Item 1", "Item 2", "Item 3"});
  gui->addVariable("Color", colval)
      ->setFinalCallback([](const Color &c) {
        std::cout << "ColorPicker Final Callback: ["
                  << c.r() << ", "
                  << c.g() << ", "
                  << c.b() << ", "
                  << c.w() << "]" << std::endl;
      });

  gui->addGroup("Other widgets");
  gui->addButton("A button", []() { std::cout << "Button pressed." << std::endl; })->setTooltip(
      "Testing a much longer tooltip, that will wrap around to new lines multiple times.");;

  screen->setVisible(true);
  screen->performLayout();
  nanoguiWindow->center();

  app.mouseCallback = [](double x, double y) {
    screen->cursorPosCallbackEvent(x, y);
  };
  app.buttonCallback = [](int button, int action, int modifiers) {
    screen->mouseButtonCallbackEvent(button, action, modifiers);
  };
  app.keyCallback = [](int key, int scancode, int action, int mods) {
    screen->keyCallbackEvent(key, scancode, action, mods);
  };
  app.charCallback = [](unsigned int codepoint) {
    screen->charCallbackEvent(codepoint);
  };
  app.dropCallback = [](int count, const char **filenames) {
    screen->dropCallbackEvent(count, filenames);
  };
  app.scrollCallback = [](double x, double y) {
    screen->scrollCallbackEvent(x, y);
  };
  app.resizeCallback = [](int width, int height) {
    screen->resizeCallbackEvent(width, height);
  };

  app.renderCallback = [&]() {
    screen->drawContents();
    screen->drawWidgets();
  };
  app.run();
  return 0;
}