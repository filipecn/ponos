#include <aergia/aergia.h>

const char *vs = "#version 440 core\n"
    // regular vertex attributes
    "in vec3 position;"
    // per instance attributes
    "in vec3 pos;"  // instance position
    "in float rad;" // instance radius
    "in vec4 col;"  // instance color
    // constants accross draw
    "uniform mat4 view_matrix;"
    "uniform mat4 projection_matrix;"
    // output to fragment shader
    "out VERTEX {"
    "vec4 color;"
    "} vertex;"
    "void main() {"
    // define model_matrix: instance transform
    "    mat4 model_matrix;"
    "    model_matrix[0] = vec4(rad, 0, 0, 0);"
    "    model_matrix[1] = vec4(0, rad, 0, 0);"
    "    model_matrix[2] = vec4(0, 0, rad, 0);"
    "    model_matrix[3] = vec4(pos.x, pos.y, pos.z, 1);"
    "    mat4 model_view_matrix = view_matrix * model_matrix;\n"
    "    gl_Position = projection_matrix * model_view_matrix * "
    "vec4(position,1);"
    "   vertex.color = col;"
    "}";
const char *fs = "#version 440 core\n"
    "in VERTEX { vec4 color; } vertex;"
    "out vec4 outColor;"
    "void main() {"
    "   outColor = vertex.color;"
    "}";

int main() {
  aergia::SceneApp<> app(800, 800);
  app.init();
  // generate base mesh
  ponos::RawMeshSPtr m(ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 3, false, false));
  // create a vertex buffer for base mesh
  aergia::SceneMesh sm(*m.get());
  // create instances container for 1000 instances (this number could be changed
  // later with resize())
  aergia::InstanceSet s(sm,
                        aergia::Shader(aergia::ShaderManager::instance().loadFromTexts(vs, nullptr, fs)),
                        2);
  // create a buffer for particles positions + sizes
  aergia::BufferDescriptor posSiz;
  posSiz.elementSize = 4;  // x y z s
  posSiz.dataType = GL_FLOAT;
  posSiz.type = GL_ARRAY_BUFFER;
  posSiz.use = GL_STREAM_DRAW;
  posSiz.addAttribute("pos", 3 /* x y z*/, 0, posSiz.dataType);
  posSiz.addAttribute("rad", 1 /* s */, 3 * sizeof(float), posSiz.dataType);
  uint pid = s.add(posSiz);
  // create a buffer for particles colors
  aergia::BufferDescriptor col;
  col.elementSize = 4;  // r g b a
  col.dataType = GL_FLOAT;
  col.type = GL_ARRAY_BUFFER;
  col.use = GL_STREAM_DRAW;
  col.addAttribute("col", 4 /* r g b a */, 0, col.dataType);
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