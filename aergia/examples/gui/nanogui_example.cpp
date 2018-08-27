// Created by filipecn on 6/12/18.
#include <aergia/aergia.h>

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

int main(int /* argc */, char ** /* argv */) {
  aergia::SceneApp<> app(800, 800, "Nanogui Example", false);
  app.addViewport2D(0, 0, 800, 800);
  std::shared_ptr<aergia::NanoGUIScreen> screen(new aergia::NanoGUIScreen());
  // Create nanogui gui
  bool enabled = true;
  FormHelper *gui = new FormHelper(dynamic_cast<nanogui::Screen *>(screen.get()));
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
  gui->addButton("A button", []() {
    std::cout << "Button pressed." << std::endl;
  })->setTooltip(
      "Testing a much longer tooltip, that will wrap around to new lines multiple times.");;

  screen->setVisible(true);
  screen->performLayout();
  nanoguiWindow->center();
  app.scene.add(new aergia::CartesianGrid(5));
  app.run();
  return 0;
}