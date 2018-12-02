#include <circe/circe.h>

int main() {
  auto objPath = std::string(ASSETS_PATH) + "/suzanne.obj";
  auto vertPath = std::string(SHADERS_PATH) + "/scene_object.vert";
  auto fragPath = std::string(SHADERS_PATH) + "/scene_object.frag";
  circe::SceneApp<> app;
  auto shader = circe::createShaderProgramPtr(
      circe::ShaderManager::instance().loadFromFiles(
          {vertPath.c_str(), fragPath.c_str()}));
  shader->addVertexAttribute("position", 0);
  shader->addVertexAttribute("normal", 1);
  shader->addVertexAttribute("texcoord", 2);
  shader->addUniform("projectionMatrix", 3);
  shader->addUniform("modelMatrix", 4);
  auto obj = circe::createSceneMeshObjectSPtr(objPath, shader);
  obj->drawCallback = [](circe::ShaderProgram *s,
                         const circe::CameraInterface *camera,
                         ponos::Transform t) {
    s->begin();
    s->setUniform("projectionMatrix",
                  ponos::transpose(camera->getProjectionTransform().matrix()));
    s->setUniform("modelMatrix",
                  ponos::transpose(camera->getViewTransform().matrix()));
  };
  circe::BVH bvh(obj);
  ponos::IndexIterator<3> it(ponos::ivec3(-10), ponos::ivec3(10));
  const char *fs = CIRCE_INSTANCES_FS;
  const char *vs = CIRCE_INSTANCES_VS;
  circe::ShaderProgram sphereShader(vs, nullptr, fs);
  sphereShader.addVertexAttribute("position", 0);
  sphereShader.addVertexAttribute("color", 1);
  sphereShader.addVertexAttribute("transform_matrix", 2);
  sphereShader.addUniform("model_view_matrix", 3);
  sphereShader.addUniform("projection_matrix", 4);
  circe::SceneMesh sphereMesh(ponos::RawMeshes::cube());
  circe::InstanceSet spheres(sphereMesh, sphereShader, it.count());
  // create a buffer for particles positions + sizes
  circe::BufferDescriptor trans = circe::create_array_stream_descriptor(16);
  trans.addAttribute("transform_matrix", 16, 0, trans.dataType);
  uint tid = spheres.add(trans);
  // create a buffer for particles colors
  circe::BufferDescriptor col =
      circe::create_array_stream_descriptor(4);  // r g b a
  col.addAttribute("color", 4, 0, col.dataType); // 4 -> r g b a
  uint colid = spheres.add(col);
  for (; it.next(); it++) {
    size_t i = it.id();
    auto color = circe::COLOR_BLACK;
    if (bvh.isInside(ponos::Point3(it()[0], it()[1], it()[2])))
      color = circe::COLOR_RED;
    auto c = spheres.instanceF(colid, i);
    c[0] = color.r;
    c[1] = color.g;
    c[2] = color.b;
    c[3] = color.a;
    c[3] = 0.4f;
    auto m = spheres.instanceF(tid, i);
    float t[16];
    (ponos::translate(ponos::vec3(it()[0], it()[1], it()[2])) *
     ponos::scale(0.1f, 0.1f, 0.1f))
        .matrix()
        .column_major(t);
    for (size_t k = 0; k < 16; k++)
      m[k] = t[k];
  }
  app.scene.add(&spheres);
  app.scene.add(new circe::CartesianGrid(10));
  app.scene.add(obj.get());
  app.run();
  return 0;
}