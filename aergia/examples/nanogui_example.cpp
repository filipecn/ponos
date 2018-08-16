// Created by filipecn on 6/12/18.
#include <aergia/aergia.h>
//#include <glad/glad.h>
//#include <Eigen/Core>
#include <nanogui/nanogui.h>
#define NANOGUI_GLAD 1

/*
    src/example3.cpp -- C++ version of an example application that shows
    how to use nanogui in an application with an already created and managed
    glfw context.

    NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
    The widget drawing code is based on the NanoVG demo application
    by Mikko Mononen.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

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

  glfwInit();

  glfwSetTime(0);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_SAMPLES, 0);
  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create a GLFWwindow object
  GLFWwindow *window = glfwCreateWindow(800, 800, "example3", nullptr, nullptr);
  if (window == nullptr) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

#if defined(NANOGUI_GLAD)
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    throw std::runtime_error("Could not initialize GLAD!");
  glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
#endif

  glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  // Create a nanogui screen and pass the glfw pointer to initialize
  screen = new Screen();
  screen->initialize(window, true);

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);
  glfwSwapInterval(0);
  glfwSwapBuffers(window);

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

  glfwSetCursorPosCallback(window,
                           [](GLFWwindow *, double x, double y) {
                             screen->cursorPosCallbackEvent(x, y);
                           }
  );

  glfwSetMouseButtonCallback(window,
                             [](GLFWwindow *, int button, int action, int modifiers) {
                               screen->mouseButtonCallbackEvent(button, action, modifiers);
                             }
  );

  glfwSetKeyCallback(window,
                     [](GLFWwindow *, int key, int scancode, int action, int mods) {
                       screen->keyCallbackEvent(key, scancode, action, mods);
                     }
  );

  glfwSetCharCallback(window,
                      [](GLFWwindow *, unsigned int codepoint) {
                        screen->charCallbackEvent(codepoint);
                      }
  );

  glfwSetDropCallback(window,
                      [](GLFWwindow *, int count, const char **filenames) {
                        screen->dropCallbackEvent(count, filenames);
                      }
  );

  glfwSetScrollCallback(window,
                        [](GLFWwindow *, double x, double y) {
                          screen->scrollCallbackEvent(x, y);
                        }
  );

  glfwSetFramebufferSizeCallback(window,
                                 [](GLFWwindow *, int width, int height) {
                                   screen->resizeCallbackEvent(width, height);
                                 }
  );

  // Game loop
  while (!glfwWindowShouldClose(window)) {
    // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
    glfwPollEvents();

    glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw nanogui
    screen->drawContents();
    screen->drawWidgets();

    glfwSwapBuffers(window);
  }

  // Terminate GLFW, clearing any resources allocated by GLFW.
  glfwTerminate();

  return 0;
}
/*



class GUI : public nanogui::Screen {
 public:
  GUI() {
nanogui::Window *window = new nanogui::Window(this, "Options");
window->
setPosition(Eigen::Vector2i(15, 15));
window->setLayout(new
nanogui::GroupLayout()
);
auto b = new nanogui::Button(window, "Load Config File");
b->setCallback([&] {
});
auto *paths = new nanogui::Widget(window);
paths->setLayout(new
nanogui::BoxLayout(nanogui::Orientation::Horizontal,
    nanogui::Alignment::Middle,
0, 2));
b = new nanogui::Button(paths, "Particles Path", ENTYPO_ICON_ARCHIVE);
b->setCallback([&]() {
auto f = nanogui::folder_dialog();
if (f.
size()
) {
particlePath->
setTooltip(f[0]);
particlePath->
setValue(f[0]
.
substr(f[0]
.
size()
- 8, 8));
}
});
particlePath = new nanogui::TextBox(paths);
paths = new nanogui::Widget(window);
paths->setLayout(new
nanogui::BoxLayout(nanogui::Orientation::Horizontal,
    nanogui::Alignment::Middle,
0, 2));
b = new nanogui::Button(paths, "Structure Path", ENTYPO_ICON_ARCHIVE);
b->setCallback([&]() {
auto f = nanogui::folder_dialog();
if (f.
size()
) {
structurePath->
setTooltip(f[0]);
structurePath->
setValue(f[0]
.
substr(f[0]
.
size()
- 8, 8));
}
});
structurePath = new nanogui::TextBox(paths);

curFrame = curSubFrame = 0;
window = new nanogui::Window(this, "Frame Control");
window->
setPosition(Eigen::Vector2i(35, 15));
window->setLayout(new
nanogui::GroupLayout()
);
paths = new nanogui::Widget(window);
paths->setLayout(new
nanogui::BoxLayout(nanogui::Orientation::Horizontal,
    nanogui::Alignment::Middle,
0, 2));
b = new nanogui::Button(paths, "", ENTYPO_ICON_CONTROLLER_JUMP_TO_START);
b->setCallback([&]() {
curFrame = std::max(0, curFrame - 1);
curSubFrame = 0;
loadFrame(curFrame, curSubFrame
);
});
b = new nanogui::Button(paths, "", ENTYPO_ICON_CHEVRON_THIN_LEFT);
b->setCallback([&]() {
curSubFrame = std::max(0, curSubFrame - 1);
loadFrame(curFrame, curSubFrame
);
});
curFrameLabel = new nanogui::TextBox(paths);
curFrameLabel->setFixedWidth(80);
curFrameLabel->setValue("");
b = new nanogui::Button(paths, "", ENTYPO_ICON_CHEVRON_THIN_RIGHT);
b->setCallback([&]() {
curSubFrame = std::max(0, curSubFrame + 1);
loadFrame(curFrame, curSubFrame
);
});
b = new nanogui::Button(paths, "", ENTYPO_ICON_CONTROLLER_NEXT);
b->setCallback([&]() {
curFrame = std::max(0, curFrame + 1);
curSubFrame = 0;
loadFrame(curFrame, curSubFrame
);
});
}
private:
void loadFrame(int frame, int subFrame) {
  curFrame = frame;
  curSubFrame = subFrame;
  std::stringstream ss;
  ss << frame << "." << subFrame;
  curFrameLabel->setValue(ss.str());
}
// simulation & rendering
int curSubFrame;
// strings
nanogui::TextBox *particlePath, *structurePath, *curFrameLabel;
int curFrame;
};

int main() {
  aergia::SceneApp<> app(800, 800, "nanogui example");
  app.init();
//  furoo::AdaptiveCenteredPic2 pic(furoo::BBox2d::squareBox(1), 4, 6);
//  ParticleSystemModel2 particles(pic.particles(), 0.001);
//  CellCenteredGraph2Model
//      model(dynamic_cast<furoo::CellGraph2 *>(pic.structure()));
//  GUI gui(pic, particles, model);
  try {
    GUI gui;
    gui.initialize(aergia::GraphicsDisplay::instance().getGLFWwindow(), true);
    gui.setVisible(true);
    gui.performLayout();
    app.mouseCallback =
        [&](double x, double y) { gui.cursorPosCallbackEvent(x, y); };
    app.buttonCallback = [&](int button, int action, int mods) {
      gui.mouseButtonCallbackEvent(button, action, mods);
    };
    app.keyCallback = [&](int key, int scancode, int action, int mods) {
      gui.keyCallbackEvent(key,
                           scancode,
                           action,
                           mods);
    };
    app.charCallback = [&](unsigned int pointcode) {
      gui.charCallbackEvent(pointcode);
    };
    app.dropCallback = [&](int count, const char **filenames) {
      gui.dropCallbackEvent(count,
                            filenames);
    };
    app.renderCallback = [&]() {
//      gui.drawContents();
      gui.drawWidgets();
    };
    app.run();
  } catch (std::string e) {
    std::cerr << e;
  }
  return 0;
}
*/