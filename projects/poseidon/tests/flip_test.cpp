#include <aergia.h>
#include <poseidon.h>
#include <ponos.h>
#include <windows.h>
#include "solver.h"
#include "flip_drawer.h"

int w = 16, h = 16;
aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
aergia::Camera2D camera;
poseidon::FLIP flip;
FLIPDrawer fd(&flip);
Solver s;
int ij(int i, int j) {return i*w + j;}

void solvePressure() {
  // construct RHS
  float scale = 1.0 / flip.dx;
  for(int i = 0; i < w; i++)
  for(int j = 0; j < h; j++) {
    s.B(ij(i, j)) = 0.f;
    if(flip.cell(i, j) == FLUID){
      // negative divergence
      s.B(ij(i, j)) = -scale * (flip.u(i + 1, j).v - flip.u(i, j).v
                              + flip.v(i, j + 1).v - flip.v(i, j).v);
      if(flip.cell(i - 1, j) == SOLID)
      s.B(ij(i, j)) -= scale * (flip.u(i, j).v - flip.usolid(i, j));
      if(flip.cell(i + 1, j) == SOLID)
      s.B(ij(i, j)) += scale * (flip.u(i + 1, j).v - flip.usolid(i + 1, j));
      if(flip.cell(i, j - 1) == SOLID)
      s.B(ij(i, j)) -= scale * (flip.v(i, j).v - flip.vsolid(i, j));
      if(flip.cell(i, j + 1) == SOLID)
      s.B(ij(i, j)) += scale * (flip.v(i, j + 1).v - flip.vsolid(i, j + 1));
    }
  }

  scale = flip.dt / (flip.rho * flip.dx * flip.dx);
  for(int i = 0; i < w; i++)
  for(int j = 0; j < h; j++) {

      s.A.coeffRef(ij(i,j),ij(i,j)) = 0.f;
      if(i > 0)     s.A.coeffRef(ij(i, j),ij(i - 1,j)) = 0.f;
      if(i < w - 1) s.A.coeffRef(ij(i, j),ij(i + 1,j)) = 0.f;
      if(j > 0)     s.A.coeffRef(ij(i, j),ij(i,j - 1)) = 0.f;
      if(j < h - 1) s.A.coeffRef(ij(i, j),ij(i,j + 1)) = 0.f;

      if(flip.cell(i, j) == FLUID) {
        s.A.coeffRef(ij(i,j),ij(i,j)) = 4.f * scale;
        if(flip.cell(i + 1, j) == SOLID)
          s.A.coeffRef(ij(i,j),ij(i,j)) -= scale;
        else if(flip.cell(i + 1, j) == FLUID)
          s.A.coeffRef(ij(i,j),ij(i + 1,j)) = -scale;

        if(flip.cell(i - 1, j) == SOLID)
          s.A.coeffRef(ij(i,j),ij(i,j)) -= scale;
        else if(flip.cell(i - 1, j) == FLUID)
          s.A.coeffRef(ij(i,j),ij(i - 1,j)) = -scale;

        if(flip.cell(i, j + 1) == SOLID)
          s.A.coeffRef(ij(i,j),ij(i,j)) -= scale;
        else if(flip.cell(i, j + 1) == FLUID)
          s.A.coeffRef(ij(i,j),ij(i,j + 1)) = -scale;

        if(flip.cell(i, j - 1) == SOLID)
          s.A.coeffRef(ij(i,j),ij(i,j)) -= scale;
        else if(flip.cell(i, j - 1) == FLUID)
          s.A.coeffRef(ij(i,j),ij(i,j - 1)) = -scale;
      }
    }

    s.solve();
    // update velocities
    scale = flip.dt / (flip.rho * flip.dx);
    for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++){
      if(flip.cell(i, j) == FLUID){
        flip.u(i, j).v -= scale * s.X(ij(i,j));
        flip.u(i + 1, j).v += scale * s.X(ij(i,j));
        flip.v(i, j).v -= scale * s.X(ij(i,j));
        flip.v(i, j + 1).v += scale * s.X(ij(i,j));
      }
    }
    flip.enforceBoundary();
}

void render(){
  gd.clearScreen(1.f, 1.f, 1.f, 0.f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  camera.look();
  fd.drawParticles();
  fd.drawMACGrid();
  fd.drawGridVelocities(flip.v, 1);
  fd.drawGridVelocities(flip.u, 0);
  fd.drawCells();

  flip.gather(flip.u, 0);
  flip.gather(flip.v, 1);
  flip.classifyCells();
  flip.addForces(flip.u, 0);
  flip.addForces(flip.v, 1);
  solvePressure();
  flip.scatter(flip.u, 0);
  flip.scatter(flip.v, 1);
  flip.advect();

  //Sleep(100);
}

int main() {
  WIN32CONSOLE();
  // init FLIP
  s.set(w * h);
  flip.set(w, h, ponos::vec2(0.f,0.f), 0.1f);
  flip.gravity = ponos::vec2(0.f, -9.8f);
  flip.dt = 0.001f;
  flip.rho = 1.f;
  for(int i = 0; i < w; i++)
    flip.isSolid(i, 0) = flip.isSolid(i, h - 1) = 1;
  for(int i = 0; i < h; i++)
    flip.isSolid(0, i) = flip.isSolid(w - 1, i) = 1;

  for (int i = 10; i < 15; ++i) {
    for (int j = 10; j < 15; ++j) {
      flip.fillCell(i, j);
    }
  }

  for (int i = 1; i < w-1; ++i) {
    for (int j = 1; j < 6; ++j) {
      flip.fillCell(i, j);
    }
  }
  // set camera
  camera.setPos(ponos::vec2(w * flip.dx / 2.f, h * flip.dx / 2.f));
  camera.setZoom(w * flip.dx / 1.5f);
  camera.resize(800,800);
  // init window
  aergia::createGraphicsDisplay(800, 800, "FLIP - 2D");
  gd.registerRenderFunc(render);
  gd.start();

  return 0;
}
