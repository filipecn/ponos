#include "algorithms/flip.h"

namespace poseidon {

  void FLIP::set(uint32_t w, uint32_t h, vec2 offset, float scale) {
    width = w;
    height = h;
    dx = scale;
    vec2 vscale = vec2(scale, scale);
    particleGrid.set(w, h, offset, vscale);
    u.set(w + 1, h, offset + vec2(-0.5f * dx, 0.f), vscale);
    u.init();
    v.set(w, h + 1, offset + vec2(0.f, -0.5f * dx), vscale);
    v.init();
    usolid.set(w + 1, h, offset + vec2(-0.5f * dx, 0.f), vscale);
    usolid.init();
    vsolid.set(w, h + 1, offset + vec2(0.f, -0.5f * dx), vscale);
    vsolid.init();
    isSolid.set(w, h, offset, vscale);
    isSolid.init();
    cell.set(w, h, offset, vscale);
    cell.init();
    vCopy[0].set(w + 1, h, offset + vec2(-0.5f * dx, 0.f), vscale);
    vCopy[0].init();
    vCopy[1].set(w, h + 1, offset + vec2(0.f, -0.5f * dx), vscale);
    vCopy[1].init();
    ps.set(w, h);
  }

  void FLIP::fillCell(uint32_t i, uint32_t j) {
    particleGrid.fillCell(i, j);
  }

  void FLIP::step() {
    gather(u, 0);
    gather(v, 1);
    classifyCells();
    addForces(u, 0);
    addForces(v, 1);
    solvePressure();
    scatter(u, 0);
    scatter(v, 1);
    advect();
  }

  void FLIP::gather(ZGrid<VelocityCell> &grid, uint32_t component) {
    grid.reset([](VelocityCell& vc){
      vc.v = 0.f;
      vc.wsum = 0.f;
    });
    uint32_t size = particleGrid.size();
    for(int i = 0; i < size; i++) {
      Point2 wp = particleGrid.getParticle(i).p;
      Point2 gp = grid.toGrid(wp);
      int xmin = static_cast<int>(gp.x);
      int ymin = static_cast<int>(gp.y);
      for(int x = xmin; x <= xmin + 1; x++)
      for(int y = ymin; y <= ymin + 1; y++) {
        if(x < 0 || x >= grid.width || y < 0 || y >= grid.height)
        continue;
        Point2 gwp = grid.toWorld(Point2(x, y));
        vec2 d = wp - gwp;
        float k = ponos::trilinear_hat_function(d.x / dx) *
        ponos::trilinear_hat_function(d.y / dx);
        grid(x, y).v += particleGrid.getParticle(i).v[component] * k;
        grid(x, y).wsum += k;
      }
    }
    grid.reset([](VelocityCell& vc){
      if(vc.wsum != 0.f) {
        vc.v = vc.v / vc.wsum;
      }
    });
    for(int i = 0; i < grid.width; i++)
      for(int j = 0; j < grid.height; j++)
        vCopy[component](i,j) = grid(i, j).v;
  }

  void FLIP::addForces(ZGrid<VelocityCell> &grid, uint32_t component) {
    grid.reset([this, component](VelocityCell& vc){
      vc.v += gravity[component] * dt;
    });
  }

  void FLIP::scatter(ZGrid<VelocityCell> &grid, uint32_t component) {
    for(int i = 0; i < grid.width; i++)
    for(int j = 0; j < grid.height; j++)
      grid(i,j).v -= vCopy[component](i, j);
    int size = particleGrid.size();
    for(int i = 0; i < size; i++) {
      Point2 p = particleGrid.getParticle(i).p;
      particleGrid.getParticleReference(i).v[component] +=
        grid.sample(p.x, p.y, [](VelocityCell& vc){ return vc.v; });
    }
  }

  void FLIP::classifyCells() {
    cell.setAll(0);
    for(int i = 0; i < cell.width; i++)
    for(int j = 0; j < cell.height; j++){
      // each cell containing at least one particle is a FLUID cell
      if(particleGrid.grid(i, j).numberOfParticles() > 0)
      cell(i, j) = FLUID;
      else if(isSolid(i, j))
      cell(i, j) = SOLID;
      else cell(i, j) = AIR;
    }
  }

  void FLIP::enforceBoundary() {
    for(int i = 0; i < cell.width; i++)
    for(int j = 0; j < cell.height; j++){
      if(cell(i, j) == SOLID){
        u(i, j).v = usolid(i, j);
        u(i + 1, j).v = usolid(i + 1, j);
        v(i, j).v = vsolid(i, j);
        v(i, j + 1).v = vsolid(i, j + 1);
      }
    }
  }

  void FLIP::solvePressure() {
    // construct RHS
    float scale = 1.0 / dx;
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++){
      if(cell(i, j) == FLUID){
        // negative divergence
        ps.rhs(i, j) = -scale * (u(i + 1, j).v - u(i, j).v
                               + v(i, j + 1).v - v(i, j).v);
        if(cell(i - 1, j) == SOLID)
        ps.rhs(i, j) -= scale * (u(i, j).v - usolid(i, j));
        if(cell(i + 1, j) == SOLID)
        ps.rhs(i, j) += scale * (u(i + 1, j).v - usolid(i + 1, j));
        if(cell(i, j - 1) == SOLID)
        ps.rhs(i, j) -= scale * (v(i, j).v - vsolid(i, j));
        if(cell(i, j + 1) == SOLID)
        ps.rhs(i, j) += scale * (v(i, j + 1).v - vsolid(i, j + 1));
      }
      else ps.rhs(i, j) = 0.0;
    }

    ps.reset();

    // set up the matrix
    scale = dt / (rho * dx * dx);
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++){
      char curCell = cell(i, j);
      if(curCell == FLUID && cell(i + 1, j) == FLUID) {
        ps.Adiag(i, j) += scale;
        ps.Adiag(i + 1, j) += scale;
        ps.Aplusi(i, j) = -scale;
      }
      else if(curCell == FLUID && cell(i + 1, j) == AIR) {
        ps.Adiag(i, j) += scale;
      }
      if(curCell == FLUID && cell(i, j + 1) == AIR) {
        ps.Adiag(i, j) += scale;
        ps.Adiag(i, j + 1) += scale;
        ps.Aplusi(i, j) = -scale;
      }
      else if(curCell == FLUID && cell(i, j + 1) == AIR) {
        ps.Adiag(i, j) += scale;
      }
    }

    ps.solve();

    DUMP_VECTOR(ps.p);
    if(std::isnan(ps.p[0]))
      exit(1);

    // update velocities
    scale = dt / (rho * dx);
    for(int i = 0; i < width; i++)
    for(int j = 0; j < height; j++){
      if(cell(i, j) == FLUID){
        u(i, j).v -= scale * ps.P(i, j);
        u(i + 1, j).v += scale * ps.P(i, j);
        v(i, j).v -= scale * ps.P(i, j);
        v(i, j + 1).v += scale * ps.P(i, j);
      }
    }
    enforceBoundary();
  }

  ponos::Point2 FLIP::newPosition(Particle2D pa) {
    ponos::Point2 e = pa.p + dt * pa.v;
    int curCell = cell.dSample(e.x, e.y, -1);
    if(curCell < 0 || curCell == CellType::SOLID) {
      while(ponos::distance(pa.p, e) > 1e-5 && (curCell < 0 || curCell == CellType::SOLID)) {
        Vector2 m = 0.5f * Vector2(pa.p.x, pa.p.y) + 0.5f * Vector2(e.x, e.y);
        curCell = cell.dSample(m.x, m.y, -1);
        if(curCell < 0 || curCell == CellType::SOLID)
        e = ponos::Point2(m.x, m.y);
        else pa.p = ponos::Point2(m.x, m.y);
        curCell = cell.dSample(e.x, e.y, -1);
      }
      e -= (dt / 10.0f) * ponos::normalize(pa.v);
    }
    return e;
  }

  void FLIP::advect() {
    int size = particleGrid.size();
    for(int i = 0; i < size; i++) {
      Particle2D pa = particleGrid.getParticle(i);
      particleGrid.setPos(i, newPosition(pa));
    }
  }

} // poseidon namespace
