#include "structures/mac_grid.h"
#include "structures/z_particle_grid.h"

#include <ponos.h>

using namespace poseidon;

enum class FLIPCellType {FLUID, AIR, SOLID};

/* simulation scene
 * Describes the scene with particles and solids.
 */
class FLIP2DScene {
	public:
		FLIP2DScene() {}
		FLIP2DScene(const char *filename);
		~FLIP2DScene();

		void load(const char *filename);
		void addForce(const ponos::vec2 &f);
		void generateParticles(int frame, ZParticleGrid2D<FLIPParticle2D> *pg);

		const std::vector<ponos::Mesh2D*>& getLiquidsGeometry() const;
		const std::vector<ponos::Mesh2D*>& getSolidsGeometry() const;
		const std::vector<ponos::Mesh2D*>& getStaticSolidsGeometry() const;

		const std::vector<FLIPParticle2D*>& getLiquidParticles() const;

		const std::vector<ponos::vec2>& getForces(int f) const;

		void buildStaticSolidLevelSet();
		void buildSolidLevelSet();

		bool isPointInsideSolid(const ponos::Point2 &p, int f, size_t *id) const;

	private:
		// Level sets
		std::shared_ptr<ponos::LevelSet2D> liquidLevelSet;
		std::shared_ptr<ponos::LevelSet2D> solidLevelSet;
		std::shared_ptr<ponos::LevelSet2D> staticSolidLevelSet;
		// external forces
		std::vector<ponos::vec2> forces;
		// meshes
		std::vector<ponos::Mesh2D*> liquidsGeometry;
		std::vector<ponos::Mesh2D*> staticSolidsGeometry;
		std::vector<ponos::Mesh2D*> solidsGeometry;
		// particles
		std::vector<FLIPParticle2D*> liquidParticles;
		std::vector<FLIPParticle2D*> solidParticles;
		std::vector<FLIPParticle2D*> staticSolidParticles;
};


class FLIP2D {
	public:
		virtual ~FLIP2D() {};

		void setup();
		bool init();
		void step();

		// grid size
		ponos::ivec3 dimensions;
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

	private:
		MacGrid2D<ponos::ZGrid> *grid[2];
};
