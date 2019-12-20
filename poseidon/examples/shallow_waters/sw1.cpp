#include <circe/circe.h>
#include <cmath>

using namespace ponos;
using namespace circe;

struct SWSolver {
  explicit SWSolver(int n, real_t d = 1.0) : N(n), dx(d / n), c0(std::sqrt(g * h0)) {
    c0 = std::sqrt(g * h0);
    h_.resize(N, 0);
    u_.resize(N, 0);
    q_[0].resize(N, vec2(0, 0));
    q_[1].resize(N, vec2(0, 0));
    init();
    A = mat2(0, 1, u0 * u0 + c0 * c0, 2 * u0);
    R = mat2(1, 1, u0 - c0, u0 + c0);
    L = mat2(u0 - c0, 0, 0, u0 + c0);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        Lminus(i, j) = std::min(L(i, j), 0.f);
    Rinv =
        mat2(u0 + c0, -1, -u0 + c0, 1) * static_cast<real_t>(1.0 / (2.0 * c0));
  }
  real_t x(int i) const { return (i - 0.5) * dx; }
  real_t h(int i) const { return h_[i]; }
  vec2 q(int i) const { return q_[current_step][i]; }
  real_t u(int i) const { return u_[i]; }
  void step(real_t dt) {
    int next_step = (current_step + 1) % 2;
    // left closed boundary
    q_[current_step][0] = q_[current_step][1];
    q_[current_step][0].y = -q_[current_step][1].y;
    // right open boundary
    q_[current_step][N-1] = q_[current_step][N-2];
    for (int i = 1; i < N - 1; ++i) {
      // Step 1: Solve at the interface between adjacent cells
      auto sum = [&](int k) -> vec2 {
        // vec2 Wimh =
        return R * Lminus * Rinv *
            (q_[current_step][k] - q_[current_step][k - 1]);
        //   Wimh = Lminus * Wimh;
        // return R * Wimh;
      };
      // Step 2: Define fluxes on cell's faces
      auto Fl = A * q_[current_step][i - 1] + sum(i);
      auto Fr = A * q_[current_step][i] + sum(i + 1);
      // Step 3: Advect
      q_[next_step][i] = q_[current_step][i] - (dt / dx) * (Fr - Fl);
    }
    current_step = next_step;
  }

  const int N = 200;
  const real_t dx;

private:
  void init() {
    for (size_t i = 0; i < h_.size(); ++i) {
      h_[i] = 2.0 * std::exp(-std::pow(x(i) - 0.5, 2) / 0.002);
      u_[i] = 0;
      q_[current_step][i].x = h_[i];
      q_[current_step][i].y = h_[i] * u_[i];
    }
  }
  real_t w1(int i) const {
    return (1.0f / (2.f * c0)) * ((u0 + c0) * h(i) - h(i) * u(i));
  }
  real_t w2(int i) const {
    return (1.0f / (2.f * c0)) * ((-u0 + c0) * h(i) + h(i) * u(i));
  }

  std::vector<real_t> h_, u_;
  std::vector<vec2> q_[2];
  mat2 R, L, A, Rinv, Lminus;
  const real_t u0 = 0.;
  const real_t h0 = 1.f;
  real_t c0;
  const real_t g = 1.f;
  int current_step = 0;
};

int main(int argc, char **argv) {
  // simulation data
  SWSolver sw(2000);
  // graphical app
  SceneApp<> app(800, 800, "", false);
  app.addViewport2D(0, 0, 800, 800);
  // set meshes to render cells
  RawMeshSPtr quadMesh(create_quad_mesh());
  SceneMesh qm(quadMesh.get());
  // set cell Q shaders
  const char *fs = CIRCE_INSTANCES_FS;
  const char *vs = CIRCE_INSTANCES_VS;
  circe::ShaderProgram quadShader(vs, nullptr, fs);
  quadShader.addVertexAttribute("position", 0);
  quadShader.addVertexAttribute("color", 1);
  quadShader.addVertexAttribute("transform_matrix", 2);
  quadShader.addUniform("model_view_matrix", 3);
  quadShader.addUniform("projection_matrix", 4);
  // set instances of cells
  circe::InstanceSet quads(qm, quadShader, sw.N);
  // create a buffer for quad transforms
  circe::BufferDescriptor transforms =
      circe::create_array_stream_descriptor(16);
  transforms.addAttribute("transform_matrix", 16, 0, transforms.dataType);
  u32 tid = quads.add(transforms);
  // create a buffer for particles colors
  circe::BufferDescriptor col =
      circe::create_array_stream_descriptor(4);  // r g b a
  col.addAttribute("color", 4, 0, col.dataType); // 4 -> r g b a
  uint colid = quads.add(col);
  quads.resize(sw.N);
  int frame = 0;
  app.renderCallback = [&]() {
    frame++;
    sw.step(0.0001);
    for (int i = 0; i < sw.N; i++) {
      auto color = circe::Color::Red();
      auto c = quads.instanceF(colid, i);
      c[0] = 0.001 * frame;
      c[1] = color.g;
      c[2] = color.b;
      c[3] = color.a;
      c[3] = 0.4;
      (ponos::translate(ponos::vec3(sw.x(i) - 0.5 * sw.dx, 0, 0)) *
          ponos::scale(sw.dx, 0.01 + sw.q(i).x, 1.f))
          .matrix()
          .column_major(quads.instanceF(tid, i));
    }
  };
  app.scene.add(&quads);
  circe::CartesianGrid grid(2);
  app.scene.add(&grid);
  app.run();
  return 0;
}