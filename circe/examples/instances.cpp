#include <circe/circe.h>

#include <memory>

int main() {
  circe::SceneApp<> app(800, 800);
  std::shared_ptr<circe::InstanceSet> spheres, quads;
  // generate a bunch of random quads
  // but now each instance has a transform matrix
  size_t n = 400;
  // generate base mesh
  ponos::RawMeshSPtr sphereMesh(
      ponos::create_icosphere_mesh(ponos::point3(), 1.f, 0, false, false));
  ponos::RawMeshSPtr quadMesh(ponos::create_quad_mesh(
      ponos::point3(0, 0, 0), ponos::point3(1, 0, 0), ponos::point3(1, 1, 0),
      ponos::point3(0, 1, 0), false, false));
  ponos::RawMeshSPtr wquadMesh(ponos::create_quad_wireframe_mesh(
      ponos::point3(0, 0, 0), ponos::point3(1, 0, 0), ponos::point3(1, 1, 0),
      ponos::point3(0, 1, 0)));
  // ponos::RawMeshSPtr circleMesh(ponos::RawMeshes::icosphere());
  ponos::RawMeshSPtr segmentMesh(
      ponos::RawMeshes::segment(ponos::point2(1, 0)));
  ponos::RawMeshSPtr cube(ponos::RawMeshes::cube());
  // circe::SceneMesh qm(*wquadMesh.get());
  circe::SceneMesh qm(segmentMesh.get());
  const char *fs = CIRCE_INSTANCES_FS;
  const char *vs = CIRCE_INSTANCES_VS;
  circe::ShaderProgram quadShader(vs, nullptr, fs);
  quadShader.addVertexAttribute("position", 0);
  quadShader.addVertexAttribute("color", 1);
  quadShader.addVertexAttribute("transform_matrix", 2);
  quadShader.addUniform("model_view_matrix", 3);
  quadShader.addUniform("projection_matrix", 4);
  quads = std::make_shared<circe::InstanceSet>(qm, quadShader, n / 2);
  {
    // create a buffer for particles positions + sizes
    circe::BufferDescriptor trans = circe::create_array_stream_descriptor(16);
    trans.addAttribute("transform_matrix", 16, 0, trans.dataType);
    uint tid = quads->add(trans);
    // create a buffer for particles colors
    circe::BufferDescriptor col =
        circe::create_array_stream_descriptor(4);  // r g b a
    col.addAttribute("color", 4, 0, col.dataType); // 4 -> r g b a
    uint colid = quads->add(col);
    quads->resize(n);
    circe::ColorPalette palette = circe::HEAT_MATLAB_PALETTE;
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
          ponos::translate(ponos::vec3(sampler.sample(
              ponos::bbox3(ponos::point3(-5, 0, 0), ponos::point3(5, 5, 5))))))
          .matrix()
          .column_major(t);
      for (size_t k = 0; k < 16; k++)
        m[k] = t[k];
    }
  }
  //  app.scene.add(spheres.get());
  app.scene.add(quads.get());
  circe::SceneObjectSPtr grid(new circe::CartesianGrid(5));
  app.scene.add(grid.get());
  app.run();
  return 0;
}
