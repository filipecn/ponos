#include <ponos.h>
#include <poseidon.h>
#include <cppunit/extensions/HelperMacros.h>

// #include "flip.h"

using namespace poseidon;

class TestFLIP : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE(TestFLIP);
  // CPPUNIT_TEST(testApplyForces);
  // CPPUNIT_TEST(testStoreParticleVelocities);
  CPPUNIT_TEST_SUITE_END();

  void testApplyForces();
  void testStoreParticleVelocities();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFLIP);
/*
void TestFLIP::testApplyForces() {
        FLIP flip;
        flip.dt = 0.001f;
        flip.dx = 0.1f;
        flip.dimensions = ponos::ivec3(10, 10, 10);
        flip.setup();
        ponos::ivec3 ijk;
        FOR_INDICES0_3D(flip.dimensions, ijk) {
                Particle *p = new Particle();
                p->position = ponos::Point3(ijk[0], ijk[1], ijk[2]);;
                p->velocity = ponos::vec3(0, 1, 2);
                flip.particleGrid->addParticle(p);
        }
        {
                auto particles = flip.particleGrid->getParticles();
                for(size_t i = 0; i < particles.size(); i++) {
                        CPPUNIT_ASSERT(particles[i]->velocity == ponos::vec3(0,
1, 2));
                }
        }
        flip.scene.addForce(ponos::vec3(2000, 0, 0));
        flip.scene.addForce(ponos::vec3(0, 1000, 0));
        flip.scene.addForce(ponos::vec3(0, 0, 3000));
        flip.applyForces();
        {
                auto particles = flip.particleGrid->getParticles();
                for(size_t i = 0; i < particles.size(); i++) {
                        CPPUNIT_ASSERT(particles[i]->velocity == ponos::vec3(2,
2, 5));
                }
        }
}

void TestFLIP::testStoreParticleVelocities() {
        FLIP flip;
        flip.dt = 0.001f;
        flip.dx = 0.1f;
        flip.dimensions = ponos::ivec3(10, 10, 10);
        flip.setup();
        ponos::ivec3 ijk;
        FOR_INDICES0_3D(flip.dimensions, ijk) {
                FLIPParticle *p = new FLIPParticle();
                p->position = ponos::Point3(ijk[0], ijk[1], ijk[2]);;
                p->velocity = ponos::vec3(0, 1, 2);
                p->p_copy = ponos::Point3(-1, -1, -1);
                p->v_copy = ponos::vec3(-1, -1, -1);
                flip.particleGrid->addParticle(p);
        }
        {
                auto particles = flip.particleGrid->getParticles();
                for(size_t i = 0; i < particles.size(); i++) {
                        FLIPParticle *p =
static_cast<FLIPParticle*>(particles[i]);
                        CPPUNIT_ASSERT(p->velocity == ponos::vec3(0, 1, 2));
                        CPPUNIT_ASSERT(p->v_copy == ponos::vec3(-1, -1, -1));
                        CPPUNIT_ASSERT(p->p_copy == ponos::Point3(-1, -1, -1));
                }
        }
        flip.storeParticleVelocities();
        {
                auto particles = flip.particleGrid->getParticles();
                for(size_t i = 0; i < particles.size(); i++) {
                        FLIPParticle *p =
static_cast<FLIPParticle*>(particles[i]);
                        CPPUNIT_ASSERT(p->velocity == p->v_copy);
                        CPPUNIT_ASSERT(p->position == p->p_copy);
                }
        }
}
*/
