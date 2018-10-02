#include <aergia/aergia.h>

int main() {
  aergia::SceneApp<> app(800, 800);
  std::shared_ptr<aergia::InstanceSet> spheres, quads;
  // generate a bunch of random quads
  // but now each instance has a transform matrix
  size_t n = 400;
  // generate base mesh
  ponos::RawMeshSPtr
      sphereMesh
      (ponos::create_icosphere_mesh(ponos::Point3(), 1.f, 0, false, false));
  ponos::RawMeshSPtr quadMesh(ponos::create_quad_mesh(ponos::Point3(0, 0, 0),
                                                      ponos::Point3(1, 0, 0),
                                                      ponos::Point3(1, 1, 0),
                                                      ponos::Point3(0, 1, 0),
                                                      false,
                                                      false));
  ponos::RawMeshSPtr wquadMesh
      (ponos::create_quad_wireframe_mesh(ponos::Point3(0, 0, 0),
                                         ponos::Point3(1, 0, 0),
                                         ponos::Point3(1, 1, 0),
                                         ponos::Point3(0, 1, 0)));
  ponos::RawMeshSPtr cube = ponos::RawMeshes::cube();
  //aergia::SceneMesh qm(*wquadMesh.get());
  aergia::SceneMesh qm(*cube.get());
  const char *qvs = "#version 440 core\n" \
"layout (location = 0) in vec3 position;" \
"layout (location = 1) in vec4 col;"  \
"layout (location = 2) in mat4 trans;" \
"layout (location = 3) uniform mat4 view_matrix;" \
"layout (location = 4) uniform mat4 projection_matrix;" \
"out VERTEX {" \
"vec4 color;" \
"} vertex;" \
"void main() {" \
"    gl_Position = projection_matrix * view_matrix * trans *" \
"vec4(position,1);" \
"   vertex.color = col;" \
"}";
  const char *fs = AERGIA_INSTANCES_FS;
  aergia::ShaderProgram quadShader(qvs, nullptr, fs);
  quadShader.addVertexAttribute("position", 0);
  quadShader.addVertexAttribute("col", 1);
  quadShader.addVertexAttribute("trans", 2);
  quadShader.addUniform("view_matrix", 3);
  quadShader.addUniform("projection_matrix", 4);
  quads.reset(new aergia::InstanceSet(qm, quadShader, n / 2));
  {
    // create a buffer for particles positions + sizes
    aergia::BufferDescriptor trans =
        aergia::create_array_stream_descriptor(16);
    trans.addAttribute("trans", 16, 0, trans.dataType);
    uint tid = quads->add(trans);
    // create a buffer for particles colors
    aergia::BufferDescriptor col =
        aergia::create_array_stream_descriptor(4); // r g b a
    col.addAttribute("col", 4, 0, col.dataType); // 4 -> r g b a
    uint colid = quads->add(col);
    quads->resize(n);
    aergia::ColorPalette palette = aergia::HEAT_MATLAB_PALETTE;
    ponos::RNGSampler sampler;
    ponos::HaltonSequence rng;
    for (size_t i = 0; i < n; i++) {
      auto color = palette((1.f * i) / n, 1.f);
      auto c = quads->instanceF(colid, i);
      c[0] = color.r;
      c[1] = color.g;
      c[2] = color.b;
      c[3] = color.a;
      c[3] = 0.4;
      auto m = quads->instanceF(tid, i);
      float t[16];
      (ponos::scale(rng.randomFloat(), rng.randomFloat(), rng.randomFloat()) *
          ponos::translate(
              ponos::vec3(sampler.sample(
                  ponos::BBox(ponos::Point3(-5, 0, 0),
                              ponos::Point3(5, 5, 5))))
          )).matrix().column_major(t);
      for (size_t k = 0; k < 16; k++)
        m[k] = t[k];
    }
  }
//  app.scene.add(spheres.get());
  app.scene.add(quads.get());
  aergia::SceneObjectSPtr grid(new aergia::CartesianGrid(5));
  app.scene.add(grid.get());
  app.run();
  return 0;
}
