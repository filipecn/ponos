#ifndef POSEIDON_ALGORITHMS_FLIP_H
#define POSEIDON_ALGORITHMS_FLIP_H

#include "elements/particle.h"
#include "structures/vdb_mac_grid.h"
#include "structures/vdb_particle_grid.h"

#include <ponos.h>

namespace poseidon {

	enum class FLIPCellType {FLUID, AIR, SOLID};
	struct FLIPParticle : public Particle {
		ponos::Point3 position;
		ponos::vec3 velocity;
		ponos::Normal normal;
		float density;
		float mass;
	};

	/* simulator
	 * Implements FLIP technique for fluid simulation.
	 */
	class FLIP {
		public:
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
			virtual ~FLIP();
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
			FLIPCellType ***cell;
			// particle grid
			VDBParticleGrid *particleGrid;

		private:
			void markCells();
			float maxDensity;
			std::vector<FLIPParticle> particles;
			VDBMacGrid *grid[2];
	};

} // poseidon namespace

#endif // POSEIDON_ALGORITHMS_FLIP_H

