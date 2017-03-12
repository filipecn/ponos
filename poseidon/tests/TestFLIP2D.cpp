#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

#include "flip_2d.h"

using namespace poseidon;

class TestFLIP2D : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestFLIP2D);
  CPPUNIT_TEST(testBuild);
  CPPUNIT_TEST(testExternalForces);
  CPPUNIT_TEST(testSendVelocitiesToGrid);
  CPPUNIT_TEST(testCopyGridVelocities);
  CPPUNIT_TEST(testMarkCells);
  CPPUNIT_TEST(testEnforceBoundaries);
  CPPUNIT_TEST(testSolver);
  CPPUNIT_TEST(testSubtractGrid);
  CPPUNIT_TEST(testComputeNegativeDivergence);
  CPPUNIT_TEST_SUITE_END();

  void testBuild();
  void testExternalForces();
  void testSendVelocitiesToGrid();
  void testCopyGridVelocities();
  void testMarkCells();
  void testEnforceBoundaries();
  void testSolver();
  void testSubtractGrid();
  void testComputeNegativeDivergence();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFLIP2D);

void TestFLIP2D::testBuild() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(8, 8);
  flip.dx = 0.3;
  flip.setup();
  for (int i = 0; i < 2; i++) {
    CPPUNIT_ASSERT(flip.grid[i]->dimensions == flip.dimensions);
    CPPUNIT_ASSERT(flip.grid[i]->v_u->getDimensions() ==
                   ponos::ivec2(flip.dimensions[0] + 1, flip.dimensions[1]));
    CPPUNIT_ASSERT(flip.grid[i]->v_v->getDimensions() ==
                   ponos::ivec2(flip.dimensions[0], flip.dimensions[1] + 1));
    CPPUNIT_ASSERT(flip.grid[i]->dx == flip.dx);
  }
}

void TestFLIP2D::testExternalForces() {
  FLIP2D<FLIPParticle2D> flip(new FLIP2DScene<FLIPParticle2D>());
  flip.dimensions = ponos::ivec2(8, 8);
  flip.dt = 0.5f;
  flip.setup();
  flip.scene->addForce(ponos::vec2(0, -10));
  ponos::HaltonSequence rngx(2), rngy(3);
  for (int i = 0; i < 100; i++)
    flip.particleGrid->add(
        ponos::Point2(rngx.randomFloat(), rngy.randomFloat()));
  flip.particleGrid->update();
  {
    ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
        *flip.particleGrid.get());
    CPPUNIT_ASSERT(it.count() == 100);
    while (it.next()) {
      (*it)->velocity = ponos::vec2(1.f);
      (*it)->type = ParticleTypes::FLUID;
      ++it;
    }
  }
  flip.particleGrid->update();
  flip.applyExternalForces();
  {
    ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
        *flip.particleGrid.get());
    CPPUNIT_ASSERT(it.count() == 100);
    while (it.next()) {
      CPPUNIT_ASSERT((*it)->velocity == ponos::vec2(1.f, 1.f - 5.f));
      ++it;
    }
  }
}

void TestFLIP2D::testSendVelocitiesToGrid() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(8, 8);
  flip.dx = 1.f;
  flip.setup();
  ponos::HaltonSequence rngx(2), rngy(3);
  for (int i = 0; i < 1000; i++)
    flip.particleGrid->add(
        ponos::Point2(rngx.randomFloat(0, 8), rngy.randomFloat(0, 8)));
  flip.particleGrid->update();
  {
    ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
        *flip.particleGrid.get());
    CPPUNIT_ASSERT(it.count() == 1000);
    while (it.next()) {
      (*it)->velocity =
          ponos::vec2(10.f * rngx.randomFloat(), 10.f * rngx.randomFloat());
      (*it)->type = ParticleTypes::FLUID;
      (*it)->mass = 1.f;
      ++it;
    }
  }
  // check grid
  {
    ponos::ivec2 ij;
    FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
    CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
    FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
    CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
  }
  flip.sendVelocitiesToGrid();
  {
    ponos::ivec2 ij;
    FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij) {
      ponos::Point2 wp = flip.grid[flip.CUR_GRID]->v_u->worldPosition(ij);
      ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
          *flip.particleGrid.get());
      float sum = 0.f, wsum = 0.f;
      while (it.next()) {
        float sd = ponos::distance2(wp, (*it)->position);
        float ssd = sqrtf(sd);
        if (ssd <= 1.5f * flip.dx) {
          float w = (*it)->mass * ponos::sharpen(sd, 1.4f);
          wsum += w;
          sum += w * (*it)->velocity.x;
        }
        ++it;
      }
      float r = (wsum > 0) ? sum / wsum : 0.f;
      CPPUNIT_ASSERT(
          IS_ZERO(r - (*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
    }
  }
  {
    ponos::ivec2 ij;
    FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij) {
      ponos::Point2 wp = flip.grid[flip.CUR_GRID]->v_v->worldPosition(ij);
      ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
          *flip.particleGrid.get());
      float sum = 0.f, wsum = 0.f;
      while (it.next()) {
        float sd = ponos::distance2(wp, (*it)->position);
        float ssd = sqrtf(sd);
        if (ssd <= 1.5f * flip.dx) {
          float w = (*it)->mass * ponos::sharpen(sd, 1.4f);
          wsum += w;
          sum += w * (*it)->velocity.y;
        }
        ++it;
      }
      float r = (wsum > 0) ? sum / wsum : 0.f;
      CPPUNIT_ASSERT(
          IS_ZERO(r - (*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
    }
  }
}

void TestFLIP2D::testCopyGridVelocities() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(80, 80);
  flip.setup();
  ponos::HaltonSequence rng(3);
  ponos::ivec2 ij;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1]) = rng.randomFloat();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1]) = rng.randomFloat();
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_u->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.COPY_GRID]->v_u)(ij[0], ij[1])));
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_v->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.COPY_GRID]->v_v)(ij[0], ij[1])));
  flip.copyGridVelocities();
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_u->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.COPY_GRID]->v_u)(ij[0], ij[1]) -
                         (*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_v->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.COPY_GRID]->v_v)(ij[0], ij[1]) -
                         (*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
}

void TestFLIP2D::testMarkCells() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(80, 80);
  flip.dx = 1.f;
  flip.setup();
  flip.markCells();
  ponos::ivec2 ij;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->cellType->getDimensions(), ij)
  CPPUNIT_ASSERT((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
                 CellType::AIR);
  for (int x = 0; x < 80; x++)
    for (int y = 0; y < 80; y++)
      flip.particleGrid->add(ponos::Point2(x, y));
  {
    ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
        *flip.particleGrid.get());
    CPPUNIT_ASSERT(it.count() == 80 * 80);
    while (it.next()) {
      (*it)->type = ParticleTypes::FLUID;
      ++it;
    }
  }
  flip.particleGrid->update();
  flip.markCells();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->cellType->getDimensions(), ij)
  CPPUNIT_ASSERT((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
                 CellType::FLUID);
  {
    ZParticleGrid2D<FLIPParticle2D>::particle_iterator it(
        *flip.particleGrid.get());
    CPPUNIT_ASSERT(it.count() == 80 * 80);
    while (it.next()) {
      (*it)->position = (*it)->position + ponos::vec2(1.1f);
      ++it;
    }
  }
  flip.particleGrid->update();
  flip.markCells();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->cellType->getDimensions(), ij)
  if (ij[0] == 0 || ij[1] == 0)
    CPPUNIT_ASSERT((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
                   CellType::AIR);
  else
    CPPUNIT_ASSERT((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
                   CellType::FLUID);
}

void TestFLIP2D::testEnforceBoundaries() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(80, 80);
  flip.dx = 1.f;
  flip.setup();
  ponos::ivec2 ij;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
  flip.markCells();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->cellType->getDimensions(), ij)
  if (ij >= ponos::ivec2(10, 10) && ij <= ponos::ivec2(20, 20))
    (*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) = CellType::SOLID;
  flip.grid[flip.CUR_GRID]->v_u->setAll(1.f);
  flip.grid[flip.CUR_GRID]->v_v->setAll(1.f);
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO(1.f - (*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_ZERO(1.f - (*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
  flip.enforceBoundaries();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->cellType->getDimensions(), ij) {
    if (ij[0] == 0)
      CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
    if (ij[0] == flip.grid[flip.CUR_GRID]->cellType->getDimensions()[0] - 1)
      CPPUNIT_ASSERT(
          IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0] + 1, ij[1])));
    if (ij[1] == 0)
      CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
    if (ij[1] == flip.grid[flip.CUR_GRID]->cellType->getDimensions()[1] - 1)
      CPPUNIT_ASSERT(
          IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1] + 1)));
    if ((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
            CellType::SOLID &&
        (*flip.grid[flip.CUR_GRID]->cellType)(ij[0] - 1, ij[1]) !=
            CellType::SOLID)
      CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1])));
    if ((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
            CellType::SOLID &&
        (*flip.grid[flip.CUR_GRID]->cellType)(ij[0] + 1, ij[1]) !=
            CellType::SOLID)
      CPPUNIT_ASSERT(
          IS_ZERO((*flip.grid[flip.CUR_GRID]->v_u)(ij[0] + 1, ij[1])));
    if ((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
            CellType::SOLID &&
        (*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1] - 1) !=
            CellType::SOLID)
      CPPUNIT_ASSERT(IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1])));
    if ((*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1]) ==
            CellType::SOLID &&
        (*flip.grid[flip.CUR_GRID]->cellType)(ij[0], ij[1] + 1) !=
            CellType::SOLID)
      CPPUNIT_ASSERT(
          IS_ZERO((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1] + 1)));
  }
}

void TestFLIP2D::testSolver() {
  FLIP2D<FLIPParticle2D> flip;
  flip.solver.set(5);
  flip.solver.setA(0, 0, 24.0);
  flip.solver.setA(0, 2, 6.0);
  flip.solver.setA(1, 1, 8.0);
  flip.solver.setA(1, 2, 2.0);
  flip.solver.setA(2, 0, 6.0);
  flip.solver.setA(2, 1, 2.0);
  flip.solver.setA(2, 2, 8.0);
  flip.solver.setA(2, 3, -6.0);
  flip.solver.setA(2, 4, 2.0);
  flip.solver.setA(3, 2, -6.0);
  flip.solver.setA(3, 3, 24.0);
  flip.solver.setA(4, 2, 2.0);
  flip.solver.setA(4, 4, 8.0);
  for (int i = 0; i < 5; i++)
    flip.solver.setB(i, 1.f);
  flip.solver.solve();
  float expected[5];
  expected[0] = 0.0104167;
  expected[1] = 0.09375;
  expected[2] = 0.125;
  expected[3] = 0.0729167;
  expected[4] = 0.09375;
  for (int i = 0; i < 5; i++)
    CPPUNIT_ASSERT(IS_EQUAL_ERROR(expected[i], flip.solver.getX(i), 1e-5));
}

void TestFLIP2D::testSubtractGrid() {
  FLIP2D<FLIPParticle2D> flip;
  flip.dimensions = ponos::ivec2(80);
  flip.setup();
  ponos::ivec2 ij;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1]) = 10.f;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1]) = 10.f;
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_u->getDimensions(), ij)
  (*flip.grid[flip.COPY_GRID]->v_u)(ij[0], ij[1]) = 5.f;
  FOR_INDICES0_2D(flip.grid[flip.COPY_GRID]->v_v->getDimensions(), ij)
  (*flip.grid[flip.COPY_GRID]->v_v)(ij[0], ij[1]) = 7.f;
  flip.subtractGrid();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_EQUAL((*flip.grid[flip.CUR_GRID]->v_u)(ij[0], ij[1]), 5.f));
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  CPPUNIT_ASSERT(IS_EQUAL((*flip.grid[flip.CUR_GRID]->v_v)(ij[0], ij[1]), 3.f));
}

void TestFLIP2D::testComputeNegativeDivergence() {
  FLIP2D<> flip;
  flip.dimensions = ponos::ivec2(80);
  flip.dx = 0.1f;
  flip.setup();
  ponos::ivec2 ij;
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_u->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_u)(ij) = ij[0];
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->v_v->getDimensions(), ij)
  (*flip.grid[flip.CUR_GRID]->v_v)(ij) = ij[1];
  flip.grid[flip.CUR_GRID]->computeNegativeDivergence();
  FOR_INDICES0_2D(flip.grid[flip.CUR_GRID]->D->getDimensions(), ij) {
    std::cout << (*flip.grid[flip.CUR_GRID]->D)(ij) << std::endl;
    CPPUNIT_ASSERT(IS_EQUAL((*flip.grid[flip.CUR_GRID]->D)(ij), -20.f));
  }
}
