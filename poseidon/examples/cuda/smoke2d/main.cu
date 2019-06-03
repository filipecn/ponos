#include "render.h"
#include <circe/circe.h>
#include <lodepng.h>
#include <poseidon/poseidon.h>

#define WIDTH 256
#define HEIGHT 256

__global__ void __applyForce(float *f, hermes::cuda::Grid2Info fInfo,
                             hermes::cuda::point2f p, float r, float v) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * fInfo.resolution.x + x;
  if (x < fInfo.resolution.x && y < fInfo.resolution.y) {
    auto wp = fInfo.toWorld(hermes::cuda::point2f(x, y));
    if (hermes::cuda::distance(p, wp) <= r)
      f[index] += v;
  }
}

void applyForce(hermes::cuda::VectorGridTexture2 &forceField,
                hermes::cuda::point2f p, float radius, hermes::cuda::vec2f v) {
  {
    auto info = forceField.u().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution.x,
                                           info.resolution.y);
    __applyForce<<<td.gridSize, td.blockSize>>>(forceField.uDeviceData(), info,
                                                p, radius, v.x);
    forceField.u().texture().updateTextureMemory();
  }
  {
    auto info = forceField.v().info();
    hermes::ThreadArrayDistributionInfo td(info.resolution.x,
                                           info.resolution.y);
    __applyForce<<<td.gridSize, td.blockSize>>>(forceField.vDeviceData(), info,
                                                p, radius, v.y);
    forceField.v().texture().updateTextureMemory();
  }
}

int main() {
  // SIM
  poseidon::cuda::GridSmokeSolver2_t<hermes::cuda::VectorGridTexture2> solver;
  solver.addScalarField();
  solver.setUIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setVIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setIntegrator(new poseidon::cuda::MacCormackIntegrator2());
  solver.setResolution(ponos::uivec2(WIDTH, HEIGHT));
  solver.setDx(1.0 / WIDTH);
  solver.init();
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.5f, 0.2f),
                                                   .1f, solver.scalarField(0));
  poseidon::cuda::GridSmokeInjector2::injectCircle(ponos::point2f(0.5f, 0.7f),
                                                   .1f, solver.scalarField(0));
  solver.scalarField(0).texture().updateTextureMemory();
  // applyForce(solver.forceFieldData(), hermes::cuda::point2f(0.5, 0.2), 0.1,
  //            hermes::cuda::vec2f(0, -100));
  // VIS
  circe::SceneApp<> app(WIDTH, HEIGHT, "", false);
  app.addViewport2D(0, 0, WIDTH, HEIGHT);
  CudaOpenGLInterop cgl(WIDTH, HEIGHT);
  circe::ScreenQuad screen;
  screen.shader->begin();
  screen.shader->setUniform("tex", 0);
  int frame = 0;
  app.renderCallback = [&]() {
    solver.stepFFT(0.001);
    // poseidon::cuda::GridSmokeInjector2::injectCircle(
    //     ponos::point2f(0.5f, 0.2f), .01f, solver.scalarField(0));
    solver.scalarField(0).texture().updateTextureMemory();
    renderDensity(WIDTH, HEIGHT, solver.scalarField(0).texture(),
                  cgl.bufferPointer<unsigned int>());
    // renderScalarGradient(WIDTH, HEIGHT, solver.pressureData().texture(),
    //                      cgl.bufferPointer(), 0, 5,
    //                      hermes::cuda::Color::blue(),
    //                      hermes::cuda::Color::red());
    renderSolids(WIDTH, HEIGHT, solver.solidData().texture(),
                 cgl.bufferPointer<unsigned int>());
    cgl.sendToTexture();
    cgl.bindTexture(GL_TEXTURE0);
    screen.render();
    // size_t w = 0, h = 0;
    // std::vector<unsigned char> data;
    // app.viewports[0].renderer->currentPixels(data, w, h);
    // unsigned error = lodepng::encode(ponos::concat("frame", frame++, ".png"),
    //                                  &data[0], static_cast<unsigned int>(w),
    //                                  static_cast<unsigned int>(h));
    // if (error)
    //   std::cout << "encoder error " << error << ": "
    //             << lodepng_error_text(error) << std::endl;
  };
  app.keyCallback = [&](int key, int scancode, int action, int modifiers) {
    if (action == GLFW_RELEASE) {
      if (key == GLFW_KEY_Q)
        app.exit();
      if (key == GLFW_KEY_SPACE) {
        solver.stepFFT(0.001);
        // std::cerr << solver.velocityData().u().texture() << std::endl;
      }
    }
  };
  bool activeForce = false;
  ponos::point2f forcePoint;
  app.buttonCallback = [&](int button, int action, int mods) {
    if (action == GLFW_PRESS) {
      activeForce = true;
      forcePoint = app.viewports[0].getMouseNPos() / 2.f + ponos::vec2f(0.5f);
    } else
      activeForce = false;
  };
  app.mouseCallback = [&](double x, double y) {
    if (activeForce) {
      auto mousePoint =
          app.viewports[0].getMouseNPos() / 2.f + ponos::vec2f(0.5f);
      hermes::cuda::point2f start(forcePoint.x, forcePoint.y);
      hermes::cuda::point2f end(mousePoint.x, mousePoint.y);
      applyForce(solver.forceFieldData(), start, 0.1,
                 hermes::cuda::normalize(end - start) * 1000.f);
      forcePoint = mousePoint;
    }
  };
  app.run();
  return 0;
}