#ifndef POSEIDON_ALGORITHMS_FLIP_H
#define POSEIDON_ALGORITHMS_FLIP_H

#include "elements/particle.h"
#include "structures/vdb_level_set.h"
#include "structures/vdb_mac_grid.h"
#include "structures/vdb_particle_grid.h"

#include <aergia.h>
#include <ponos.h>

namespace poseidon {

	enum class FLIPCellType {FLUID, AIR, SOLID};

	struct FLIPParticle : public Particle {
		ponos::Normal normal;
		int meshId;

		ponos::Point3 p_copy;
		ponos::vec3 v_copy;
	};


	/* simulation scene
	 * Describes the scene with particles and solids.
	 */
	class FLIPScene {
		public:
			FLIPScene() {}
			FLIPScene(const char *filename);
			~FLIPScene();

			void load(const char *filename);
			void addForce(const ponos::vec3 &f);
			void generateParticles(int frame, VDBParticleGrid *pg);

			const std::vector<aergia::Mesh*>& getLiquidsGeometry() const;
			const std::vector<aergia::Mesh*>& getSolidsGeometry() const;
			const std::vector<aergia::Mesh*>& getStaticSolidsGeometry() const;

			const std::vector<FLIPParticle*>& getLiquidParticles() const;

			const std::vector<ponos::vec3>& getForces(int f) const;

			void buildStaticSolidLevelSet();
			void buildSolidLevelSet();

			bool isPointInsideSolid(const ponos::Point3 &p, int f, size_t *id) const;

		private:
			// Level sets
			std::shared_ptr<VDBLevelSet> liquidLevelSet;
			std::shared_ptr<VDBLevelSet> solidLevelSet;
			std::shared_ptr<VDBLevelSet> staticSolidLevelSet;
			// external forces
			std::vector<ponos::vec3> forces;
			// meshes
			std::vector<aergia::Mesh*> liquidsGeometry;
			std::vector<aergia::Mesh*> staticSolidsGeometry;
			std::vector<aergia::Mesh*> solidsGeometry;
			// particles
			std::vector<FLIPParticle*> liquidParticles;
			std::vector<FLIPParticle*> solidParticles;
			std::vector<FLIPParticle*> staticSolidParticles;
	};

	/* simulator
	 * Implements FLIP technique for fluid simulation.
	 */
	class FLIP {
		public:
			virtual ~FLIP();
			/* allocation
			 * Allocate memory for internal structures. Must be called after dimensions are set. (before adding particles if adding manually)
			 */
			void setup();
			/* init
			 * Initialize structures. Should be called before simulation, but after scene setup.
			 * **note:** assumes all public fields have been previously set.
			 *
			 * @return **true** if success
			 */
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

			// cell type
			ponos::RegularGrid<FLIPCellType> cell;
			// particle grid
			VDBParticleGrid *particleGrid;
			// scene
			FLIPScene scene;

			void markCells();
			void storeParticleVelocities();
			void applyForces();
			void updateMACGrid();
			void enforceBoundaries();
			void project();
			void extrapolateVelocities();

			float maxDensity;

		private:
			VDBMacGrid *grid[2];
			ponos::RegularGrid<float> L;
	};

} // poseidon namespace

#endif // POSEIDON_ALGORITHMS_FLIP_H

