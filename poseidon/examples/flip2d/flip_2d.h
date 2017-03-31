#include "algorithms/fast_sweep.h"
#include "algorithms/solver.h"
#include "structures/mac_grid.h"
#include "structures/z_particle_grid.h"

#include <ponos.h>

using namespace poseidon;

/** \brief simulation scene description
 *
 * Describes the scene with particles and solids.
 */
template <typename ParticleType = FLIPParticle2D> class FLIP2DScene {
public:
  FLIP2DScene() {}
  FLIP2DScene(const char *filename, ZParticleGrid2D<ParticleType> *_pg);

  ~FLIP2DScene();

  void load(const char *filename);
  void addForce(const ponos::vec2 &f);
  void generateParticles(int frame);
  void fillCell(int frame, ponos::Mesh2D *liquidGeometry,
                const ponos::ivec2 &ij);
  const std::vector<ponos::Mesh2D *> &getLiquidsGeometry() const;
  const std::vector<ponos::Mesh2D *> &getSolidsGeometry() const;
  const std::vector<ponos::Mesh2D *> &getStaticSolidsGeometry() const;
  const std::vector<ParticleType *> &getLiquidParticles() const;
  const std::vector<ponos::vec2> &getForces(int f) const;

  void buildStaticSolidLevelSet();
  void buildSolidLevelSet();

  bool isPointInsideSolid(const ponos::Point2 &p, int f, size_t *id) const;

private:
  // particle grid
  std::shared_ptr<ZParticleGrid2D<ParticleType>> pg;
  // Level sets
  std::shared_ptr<ponos::LevelSet2D> liquidLevelSet;
  std::shared_ptr<ponos::LevelSet2D> solidLevelSet;
  std::shared_ptr<ponos::LevelSet2D> staticSolidLevelSet;
  // external forces
  std::vector<ponos::vec2> forces;
  // meshes
  std::vector<ponos::Mesh2D *> liquidsGeometry;
  std::vector<ponos::Mesh2D *> staticSolidsGeometry;
  std::vector<ponos::Mesh2D *> solidsGeometry;
  // particles
  std::vector<ParticleType *> liquidParticles;
  std::vector<ParticleType *> solidParticles;
  std::vector<ParticleType *> staticSolidParticles;
  // random generator
  ponos::HaltonSequence rng;
};

template <typename ParticleType = FLIPParticle2D> class FLIP2D {
  friend class TestFLIP2D;

public:
  virtual ~FLIP2D(){};
  FLIP2D(const char *filename = nullptr);
  FLIP2D(FLIP2DScene<ParticleType> *s);

  MacGrid2D<ponos::ZGrid> *getGrid() { return grid[CUR_GRID]; }
  void loadScene(const char *filename);
  void setup();
  bool init();
  void step();

  // grid size
  ponos::ivec2 dimensions;
  // density
  float density;
  // ratio between flip and pic
  float pic_flip_ratio;
  // number of sub cells
  int subcell;
  // time step size
  float dt;
  // cell size
  float dx;
  // current step
  int curStep;

  std::shared_ptr<ZParticleGrid2D<ParticleType>> particleGrid;
  std::shared_ptr<FLIP2DScene<ParticleType>> scene;

private:
  void markCells();
  void computeDensity();
  void applyExternalForces();
  void sendVelocitiesToGrid();
  void copyGridVelocities();
  void enforceBoundaries();
  int matrixIndex(int i, int j);
  void project();
  void extrapolateVelocity();
  void subtractGrid();
  void solvePICFLIP();
  void advectParticles();
  void resampleParticles();

  float maxDensity;
  size_t COPY_GRID, CUR_GRID;
  MacGrid2D<ponos::ZGrid> *grid[2];
  ponos::ZGrid<float> xDistances, yDistances, usolid, vsolid;
  Solver solver;
};

#include "flip_2d.inl"
