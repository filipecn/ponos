#include "flip.h"

namespace poseidon {

	FLIP::~FLIP() {
		ponos::ivec2 d = dimensions.xy();
		ponos::ivec2 ij;
		FOR_INDICES0_2D(d, ij)
			delete[] cell[ij[0]][ij[1]];
		int i;
		FOR_LOOP(i, 0, dimensions[0])
			delete[] cell[i];
	}

	void FLIP::setup() {
		particleGrid = new VDBParticleGrid(dimensions, dx, ponos::vec3());
		cell = new FLIPCellType**[dimensions[0]];
		int i;
		FOR_LOOP(i, 0, dimensions[0])
			cell[i] = new FLIPCellType*[dimensions[1]];
		ponos::ivec2 d = dimensions.xy();
		ponos::ivec2 ij;
		FOR_INDICES0_2D(d, ij)
			cell[ij[0]][ij[1]] = new FLIPCellType[dimensions[2]];
	}

	bool FLIP::init() {
		markCells();
		return true;
	}

	void FLIP::markCells() {
		ponos::ivec3 ijk;
		FOR_INDICES0_3D(dimensions, ijk) {
			cell[ijk[0]][ijk[1]][ijk[2]] = FLIPCellType::AIR;
			if(particleGrid->particleCount(ijk))
				cell[ijk[0]][ijk[1]][ijk[2]] = FLIPCellType::FLUID;
		}
	}

} // poseidon namespace
