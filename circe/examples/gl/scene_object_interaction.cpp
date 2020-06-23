#include <circe/circe.h>
#include <sstream>

using namespace circe::gl;

class SceneObjectInteraction : public BaseApp {
public:
  SceneObjectInteraction() : BaseApp(800, 800) {
    grid = std::make_unique<CartesianGrid>(5);
    this->app_->scene.add(grid.get());
    circle_mesh = ponos::RawMeshes::circle();
    mesh = std::make_unique<SceneMeshObject>(circle_mesh.get());
    this->app_->scene.add(mesh.get());
  }

  void render(circe::CameraInterface *camera) override {
    ImGui::Begin("Another Window", &show_another_window);
    std::stringstream ss;
    ss << this->last_FPS_ << "fps" << std::endl;
    ss << "View Matrix\n";
    ss << camera->getViewTransform().matrix() << std::endl;
    ss << "Model Matrix\n";
    ss << camera->getModelTransform().matrix() << std::endl;
    ss << "Projection Matrix\n";
    ss << camera->getProjectionTransform().matrix() << std::endl;
    ss << "MVP Matrix\n";
    ss << camera->getTransform().matrix() << std::endl;
    ss << GraphicsDisplay::instance().getMousePos() << std::endl;
    ImGui::Text(ss.str().c_str());
    ImGui::End();
  }

  bool show_another_window = false;
  std::unique_ptr<CartesianGrid> grid;
  ponos::RawMeshSPtr circle_mesh;
  std::unique_ptr<SceneMeshObject> mesh;
};

int main() {
  SceneObjectInteraction example;
  return example.run();
}