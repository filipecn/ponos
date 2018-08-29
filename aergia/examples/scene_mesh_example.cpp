// Created by filipecn on 2/25/18.
#include <aergia/aergia.h>

const char *vs = "#version 440 core\n"
    // regular vertex attributes
    "in vec3 position;"
    "in vec3 normal;"
    "in vec2 texcoord;"
    // per instance attributes
    "uniform mat4 proj;"
    "uniform mat4 model;"
    //       "uniform vec3 ldir;"
    // output to fragment shader
    "out VERTEX {"
    //  "vec4 color;"
    //       "vec3 normal;"
    "vec2 uv;"
    "} vertex;"
    "void main() {"
    "    gl_Position = proj * model * vec4(position,1);"
    //   "    vertex.normal = normalize(model * vec4(normal,
    //   0)).xyz;" "    float intensity = max(dot(vertex.normal,
    //   (model * vec4(ldir,0)).xyz), 0.0);"
    // "    vertex.color =   vec4(0,1,1,0.5) * intensity;"
    "    vertex.uv = texcoord;"
    "}";
const char *fs = "#version 440 core\n"
    "uniform sampler2D tex;"
    "in VERTEX {"
    // " vec4 color;"
    //" vec3 normal; "
    "vec2 uv; } vertex;"
    "out vec4 outColor;"
    "void main() {"
    //   "   outColor = vertex.color;"
    " outColor = texture2D(tex, vertex.uv);"
    //"outColor = vec4(vertex.uv.x,0,vertex.uv.y,1);"
    "}";

int main() {
  aergia::SceneApp<> app(800, 800, "", false);
  app.init();
  app.addViewport(0, 0, 800, 800);
  ponos::RawMesh mesh;
  // aergia::loadOBJ("/mnt/windows/Users/fuiri/Desktop/dragon.obj", &mesh);
  aergia::loadOBJ("/mnt/windows/Projects/ponos/aergia/examples/assets/torusknot.obj", &mesh);
  mesh.apply(ponos::scale(0.1, 0.1, 0.1));
  auto texture = aergia::ImageTexture::checkBoard(64, 64);
  std::cerr << texture;
  //ponos::RawMeshSPtr m(ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 3, true, false));
  // create a vertex buffer for base mesh
  //aergia::SceneMesh sm(*m.get());
  aergia::SceneMesh smesh(mesh);
  auto s = aergia::ShaderProgram(aergia::ShaderManager::instance().loadFromTexts(vs, nullptr, fs));
  // create a buffer for particles positions + sizes
  app.viewports[0].renderCallback = [&]() {
    smesh.bind();
    smesh.vertexBuffer()->locateAttributes(s);
    texture.bind(GL_TEXTURE0);
    s.begin();
    s.setUniform("proj", ponos::transpose(aergia::glGetProjectionTransform().matrix()));
    s.setUniform("model", ponos::transpose(aergia::glGetModelviewTransform().matrix()));
    s.setUniform("tex", 0);
//    s.setUniform("ldir", ponos::vec3(1,0,0));
    aergia::CHECK_GL_ERRORS;
    auto ib = smesh.indexBuffer();
    glDrawElements(ib->bufferDescriptor.elementType,
                   ib->bufferDescriptor.elementCount *
                       ib->bufferDescriptor.elementSize,
                   ib->bufferDescriptor.dataType, 0);
    aergia::CHECK_GL_ERRORS;
    s.end();
  };
  aergia::SceneObjectSPtr grid(new aergia::CartesianGrid(5));
  app.scene.add(grid.get());
  app.run();
  return 0;
}
