// Created by filipecn on 3/15/18.
#include <aergia/aergia.h>

const char *vs = "#version 440 core\n"
    // regular vertex attributes
    "in vec3 position;"
    "in vec3 normal;"
    "in vec3 texcoord;"
    // per instance attributes
    "uniform mat4 proj;"
    "uniform mat4 model;"
    // output to fragment shader
    "out VERTEX {"
    "vec4 color;"
    "vec3 normal;"
    "} vertex;"
    "void main() {"
    "    gl_Position = proj * model * vec4(position,1);"
    "    vertex.normal = normalize(model * vec4(normal, 0)).xyz;"
    "    vertex.color = vec4(texcoord,0.9);"
    "}";
const char *fs = "#version 440 core\n"
    "in VERTEX { "
    "vec4 color; "
    "vec3 normal; } vertex;"
    "out vec4 outColor;"
    "void main() {"
    "   outColor = vertex.color;"
    "}";

size_t curTet;

int main() {
  aergia::SceneApp<> app(800, 800);
  app.init();
  ponos::RawMesh dragonSurface, dragonVolume;
  //aergia::loadOBJ("/home/filipecn/Desktop/ddragon.obj", &dragonSurface);
  //aergia::loadOBJ("C:/Users/fuiri/Desktop/dragon.obj", &dragonSurface);
   aergia::loadOBJ("/mnt/windows/Projects/ponos/aergia/examples/assets/torusknot.obj", &dragonSurface);
  int minIndex = 1 << 20;
  for (auto i : dragonSurface.indices)
    minIndex = std::min(minIndex, i.positionIndex);
  std::cerr << "min index " << minIndex << std::endl;
  dragonSurface.apply(ponos::scale(0.1, 0.1, 0.1));
  ponos::tetrahedralize(&dragonSurface, &dragonVolume);
  ponos::TMesh<> tmesh(dragonVolume);
  ponos::fastMarchTetraheda(&tmesh, {0}, &tmesh);
  // color vertices
  aergia::ColorPalette palette = aergia::HEAT_MATLAB_PALETTE;
  float m = INFINITY, M = -INFINITY;
  for (auto v : tmesh.vertices) {
    m = std::min(m, v.data);
    M = std::max(M, v.data);
  }
  {
    dragonSurface.texcoords.clear();
    dragonSurface.texcoordDescriptor.count = dragonSurface.positionDescriptor.count;
    dragonSurface.texcoordDescriptor.elementSize = 3;
    for (size_t i = 0; i < dragonSurface.indices.size(); i++) {
      dragonSurface.indices[i].texcoordIndex = dragonSurface.indices[i].positionIndex;
    }
    for (size_t i = 0; i < dragonSurface.texcoordDescriptor.count; i++) {
      auto c = palette(ponos::smoothStep(tmesh.vertices[i].data, m, M), 1.f);
      dragonSurface.texcoords.emplace_back(c.r);
      dragonSurface.texcoords.emplace_back(c.g);
      dragonSurface.texcoords.emplace_back(c.b);
      std::cerr << c.r << " ";
      std::cerr << c.g << " ";
      std::cerr << c.b << std::endl;
    }
  }
  aergia::SceneMesh smesh(dragonSurface);
  auto s = aergia::ShaderProgram(aergia::ShaderManager::instance().loadFromTexts(vs, nullptr, fs));
  //std::shared_ptr<aergia::Text> text;
  //text.reset(new aergia::Text("/mnt/windows/Windows/Fonts/arial.ttf"));
  app.viewports[0].renderCallback = [&]() {
    smesh.bind();
    smesh.vertexBuffer()->locateAttributes(s);
    s.begin();
    s.setUniform("proj", aergia::glGetProjectionTransform().matrix());
    s.setUniform("model", aergia::glGetModelviewTransform().matrix());
    aergia::CHECK_GL_ERRORS;
    auto ib = smesh.indexBuffer();
    glDrawElements(ib->bufferDescriptor.elementType,
                   ib->bufferDescriptor.elementCount *
                       ib->bufferDescriptor.elementSize,
                   ib->bufferDescriptor.dataType, 0);
    s.end();
    for (auto v : tmesh.vertices) {
      GL_DRAW_POINTS(5.f,
                     aergia::glColor(palette(ponos::smoothStep(v.data, m, M), 1.f));
                         aergia::glVertex(v.position);
      )
    }
  };
  aergia::TMeshModel<> mm(tmesh);
  // app.scene.add(&mm);
  mm.facesColor = aergia::COLOR_TRANSPARENT;
  app.run();
  return 0;
}
