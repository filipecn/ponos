#include <aergia/aergia.h>

const char *vs = AERGIA_INSTANCES_VS;
const char *fs = AERGIA_INSTANCES_FS;
int main() {
  aergia::SceneApp<> app(800, 800);
  // generate base mesh
  ponos::RawMeshSPtr m(ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 3, false, false));
  // create a vertex buffer for base mesh
  aergia::SceneMesh sm(*m.get());
  // create instances container for 1000 instances (this number could be changed
  // later with resize())
  aergia::ShaderProgram shader(vs, nullptr, fs);
  shader.addVertexAttribute("position", 0);
  shader.addVertexAttribute("pos", 1);
  shader.addVertexAttribute("scale", 2);
  shader.addVertexAttribute("col", 3);
  shader.addUniform("view_matrix", 4);
  shader.addUniform("projection_matrix", 5);
  aergia::InstanceSet s(sm, shader, 2);
  // create a buffer for particles positions + sizes
  aergia::BufferDescriptor posSiz =
      aergia::create_array_stream_descriptor(4); // x y z s
  posSiz.addAttribute("pos", 3, 0, posSiz.dataType); // 3 -> x y z
  posSiz.addAttribute("scale", 1, 3 * sizeof(float), posSiz.dataType); // 1 -> s
  uint pid = s.add(posSiz);
  // create a buffer for particles colors
  aergia::BufferDescriptor col =
      aergia::create_array_stream_descriptor(4); // r g b a
  col.addAttribute("col", 4, 0, col.dataType); // 4 -> r g b a
  uint colid = s.add(col);
  // define instance data
  {
    auto v = s.instanceF(pid, 0);
    v[0] = v[1] = v[2] = 0.f;
    v[3] = 1.f;
    auto c = s.instanceF(colid, 0);
    c[0] = c[3] = 1.f;
    c[1] = c[2] = 0.f;
  }
  {
    auto v = s.instanceF(pid, 1);
    v[0] = v[1] = 2.f;
    v[2] = 1.f;
    v[3] = 2.f;
    auto c = s.instanceF(colid, 1);
    c[0] = c[2] = .5f;
    c[1] = c[3] = 1.f;
  }
  app.scene.add(&s);
  aergia::SceneObjectSPtr grid(new aergia::CartesianGrid(5));
  app.scene.add(grid.get());
  app.run();
  return 0;
}
