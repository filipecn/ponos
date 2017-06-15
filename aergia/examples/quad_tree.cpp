#include <aergia.h>
using namespace aergia;
using namespace ponos;

SceneApp<> app(800, 800, "Hello QuadTree!", false);

int main() {
  WIN32CONSOLE();
  app.init();
  typedef QuadTree<NeighbourQuadTreeNode<float>> TreeType;
  BBox2D region = make_unit_bbox2D();
  TreeType tree(region, [](TreeType::Node &node) -> bool {
    if (node.id % 2)
      return false;
    if (node.level() < 3)
      return true;
    return false;
  });
  buildLeafNeighbourhood(&tree);
  QuadTreeModel<TreeType> model(&tree);
  Text text("C:/Windows/Fonts/Arial.ttf");
  model.drawCallback = [&text](const typename TreeType::Node &n) {
    for (auto neighbour : n.data.neighbours) {
      if (neighbour->data.isPhantom)
        glColor4f(0, 0, 1, 0.1);
      else
        glColor4f(0, 0, 1, 0.5);
      glBegin(GL_LINES);
      glVertex(n.region().center());
      glVertex(neighbour->region().center());
      glEnd();
    }
    std::ostringstream stringStream;
    stringStream << n.id;
    text.render(stringStream.str(), glGetMVPTransform()(n.region().center()),
                0.4f, Color(1.f, 0.f, 0.f, 0.5f));
  };
  app.addViewport2D(0, 0, 800, 800);
  app.getCamera<Camera2D>(0)->fit(region, 1.2f);
  app.scene.add(&model);
  app.run();
  return 0;
}
