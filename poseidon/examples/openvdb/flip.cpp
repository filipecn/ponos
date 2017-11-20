#include "flip.h"

#include <openvdb/math/ConjGradient.h>

namespace poseidon {

	FLIPScene::FLIPScene(const char *filename) {
		load(filename);
	}

	FLIPScene::~FLIPScene() {}

	void FLIPScene::load(const char *filename) {
		FILE *fp = fopen(filename, "r");
		if(!fp)
			return;
		char line[1000];
		// read solids
		while(fgets(line, 1000, fp) != nullptr && line[0] == '#');
		int n = 0;
		sscanf(line, "%d", &n);
		for(int i = 0; i < n; i++) {
			fgets(line, 1000, fp);
			line[strlen(line) - 1] = 0;
			std::cout << "loading " << line << std::endl;
			aergia::RawMesh *m = new aergia::RawMesh();
			aergia::loadOBJ(line, m);
			fgets(line, 1000, fp);
			float sx, sy, sz, ox, oy, oz;
			sscanf(line, "%f %f %f %f %f %f", &sx, &sy, &sz, &ox, &oy, &oz);
			printf("%f %f %f %f %f %f\n", sx, sy, sz, ox, oy, oz);
			staticSolidsGeometry.emplace_back(new aergia::Mesh(m, ponos::translate(ponos::vec3(ox, oy, oz)) * ponos::scale(sz, sy, sz)));
		}
		// read liquids
		while(fgets(line, 1000, fp) != nullptr && line[0] == '#');
		sscanf(line, "%d", &n);
		for(int i = 0; i < n; i++) {
			fgets(line, 1000, fp);
			line[strlen(line) - 1] = 0;
			std::cout << "loading " << line << std::endl;
			aergia::RawMesh *m = new aergia::RawMesh();
			aergia::loadOBJ(line, m);
			fgets(line, 1000, fp);
			float sx, sy, sz, ox, oy, oz;
			sscanf(line, "%f %f %f %f %f %f", &sx, &sy, &sz, &ox, &oy, &oz);
			printf("%f %f %f %f %f %f\n", sx, sy, sz, ox, oy, oz);
			liquidsGeometry.emplace_back(new aergia::Mesh(m, ponos::translate(ponos::vec3(ox, oy, oz)) * ponos::scale(sz, sy, sz)));
		}
		fclose(fp);
	}

	void FLIPScene::addForce(const ponos::vec3 &f) {
		forces.emplace_back(f);
	}

	void FLIPScene::generateParticles(int frame, VDBParticleGrid *pg) {
		// generate liquid particles
		for(size_t l = 0; l < liquidsGeometry.size(); l++) {
			ponos::BBox bbox = liquidsGeometry[l]->getBBox();
			ponos::ivec3 lmin = ponos::floor(ponos::vec3(pg->toGrid(bbox.pMin)));
			ponos::ivec3 lmax = ponos::ceil(ponos::vec3(pg->toGrid(bbox.pMax)));
			lmin = ponos::max(lmin, ponos::ivec3());
			lmax = ponos::min(lmax, pg->dimensions + ponos::ivec3(1.0f));
			ponos::parallel_for(lmin[0], lmax[0] + 1, [=](size_t start, size_t end) {
					ponos::ivec3 m(start, lmin[1], lmin[2]);
					ponos::ivec3 M(end, lmax[1] + 1, lmax[2] + 1);
					ponos::ivec3 ijk;
					FOR_INDICES3D(m, M, ijk) {
						ponos::Point3 wp = pg->indexToWorld(ijk);
						if(liquidsGeometry[l]->intersect(wp)){
							size_t solidGeomID = 0;
							if(!isPointInsideSolid(wp, frame, &solidGeomID)){
								FLIPParticle* p = new FLIPParticle;
								p->position = wp;
								p->velocity = ponos::vec3();
								p->normal = ponos::Normal();
								p->density = 10.0f;
								p->type = ParticleType::FLUID;
								p->mass = 1.0f;
								p->invalid = false;
								liquidParticles.emplace_back(p);
							}
						}
					}
			});
		}
		// generate solid particles
		for(size_t s = 0; s < solidsGeometry.size(); s++) {
			ponos::BBox bbox = solidsGeometry[s]->getBBox();
			ponos::ivec3 lmin = ponos::floor(ponos::vec3(bbox.pMin));
			ponos::ivec3 lmax = ponos::ceil(ponos::vec3(bbox.pMax));
			lmin = ponos::max(lmin, ponos::ivec3());
			lmax = ponos::min(lmax, pg->dimensions + ponos::ivec3(1.0f));
			ponos::parallel_for(lmin[0], lmax[0] + 1, [=](size_t start, size_t end) {
					ponos::ivec3 m(start, lmin[1], lmin[2]);
					ponos::ivec3 M(end, lmax[1] + 1, lmax[2] + 1);
					ponos::ivec3 ijk;
					FOR_INDICES3D(m, M, ijk) {
						ponos::Point3 wp = pg->indexToWorld(ijk);
						if(solidsGeometry[s]->intersect(wp)){
								FLIPParticle* p = new FLIPParticle;
								p->position = wp;
								p->velocity = ponos::vec3();
								p->normal = ponos::Normal();
								p->density = 10.0f;
								p->type = ParticleType::SOLID;
								p->mass = 1.0f;
								p->invalid = false;
								solidParticles.emplace_back(p);
						}
					}
			});
		}
		// generate static solid particles
		for(size_t s = 0; s < staticSolidsGeometry.size(); s++) {
			ponos::BBox bbox = staticSolidsGeometry[s]->getBBox();
			ponos::ivec3 lmin = ponos::floor(ponos::vec3(bbox.pMin));
			ponos::ivec3 lmax = ponos::ceil(ponos::vec3(bbox.pMax));
			lmin = ponos::max(lmin, ponos::ivec3());
			lmax = ponos::min(lmax, pg->dimensions + ponos::ivec3(1.0f));
			std::cout << bbox.pMin << bbox.pMax;
			std::cout << "generating static solid particles " << lmin << lmax;
			ponos::parallel_for(lmin[0], lmax[0] + 1, [=](size_t start, size_t end) {
					ponos::ivec3 m(start, lmin[1], lmin[2]);
					ponos::ivec3 M(end, lmax[1] + 1, lmax[2] + 1);
					ponos::ivec3 ijk;
					FOR_INDICES3D(m, M, ijk) {
						ponos::Point3 wp = pg->indexToWorld(ijk);
						if(staticSolidsGeometry[s]->intersect(wp)){
								FLIPParticle* p = new FLIPParticle;
								p->position = wp;
								p->velocity = ponos::vec3();
								p->normal = ponos::Normal();
								p->density = 10.0f;
								p->type = ParticleType::SOLID;
								p->mass = 1.0f;
								p->invalid = false;
								solidParticles.emplace_back(p);
						}
					}
			});
		}

		for(auto p : liquidParticles)
			pg->addParticle(p);
		for(auto p : solidParticles)
			pg->addParticle(p);
		for(auto p : staticSolidParticles)
			pg->addParticle(p);
		pg->init();
	}

	const std::vector<aergia::Mesh*>& FLIPScene::getLiquidsGeometry() const {
		return liquidsGeometry;
	}

	const std::vector<aergia::Mesh*>& FLIPScene::getStaticSolidsGeometry() const {
		return staticSolidsGeometry;
	}

	const std::vector<aergia::Mesh*>& FLIPScene::getSolidsGeometry() const {
		return solidsGeometry;
	}

	const std::vector<FLIPParticle*>& FLIPScene::getLiquidParticles() const {
		return liquidParticles;
	}

	void FLIPScene::buildStaticSolidLevelSet() {
		for(size_t i = 0; i < staticSolidsGeometry.size(); i++) {
			if(i == 0)
				staticSolidLevelSet.reset(new VDBLevelSet(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
			else staticSolidLevelSet->merge(new VDBLevelSet(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
		}
	}

	void FLIPScene::buildSolidLevelSet() {
		for(size_t i = 0; i < solidsGeometry.size(); i++) {
			if(i == 0)
				staticSolidLevelSet.reset(new VDBLevelSet(solidsGeometry[i]->getMesh(), solidsGeometry[i]->getTransform()));
			else staticSolidLevelSet->merge(new VDBLevelSet(solidsGeometry[i]->getMesh(), solidsGeometry[i]->getTransform()));
		}

		if(solidsGeometry.size())
			solidLevelSet->copy(staticSolidLevelSet.get());
		else
			solidLevelSet->merge(staticSolidLevelSet.get());
	}

	bool FLIPScene::isPointInsideSolid(const ponos::Point3 &p, int f, size_t *id) const {
		for(size_t i = 0; i < solidsGeometry.size(); i++){
			if(solidsGeometry[i]->intersect(p)) {
				*id = i;
				return true;
			}
		}
		return false;
	}

	const std::vector<ponos::vec3>& FLIPScene::getForces(int f) const {
		return forces;
	}

	FLIP::~FLIP() {
		delete particleGrid;
	}

	void FLIP::setup() {
		particleGrid = new VDBParticleGrid(dimensions, dx, ponos::vec3());
		cell.set(dimensions, FLIPCellType::SOLID);
		L.set(dimensions, 0.f);
	}

	bool FLIP::init() {
		scene.buildStaticSolidLevelSet();
		VDBParticleGrid pg(ponos::ivec3(10, 10, 10), dx, ponos::vec3());
		ponos::ivec3 ijk;
		FOR_INDICES0_3D(ponos::ivec3(10, 10, 10), ijk) {
			pg.addParticle(ijk, ponos::Point3(-0.25, -0.25, -0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3(-0.25, -0.25,  0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3(-0.25,  0.25, -0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3(-0.25,  0.25,  0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3( 0.25, -0.25, -0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3( 0.25, -0.25,  0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3( 0.25,  0.25, -0.25), ponos::vec3());
			pg.addParticle(ijk, ponos::Point3( 0.25,  0.25,  0.25), ponos::vec3());
		}
		maxDensity = 1.f;
		//maxDensity = pg.computeDensity(density, maxDensity);
		scene.generateParticles(0, particleGrid);
		markCells();
		return true;
	}

	void FLIP::markCells() {
		ponos::ivec3 ijk;
		auto particles = particleGrid->getParticles();
		FOR_INDICES0_3D(dimensions, ijk) {
			cell(ijk) = FLIPCellType::AIR;
			if(particleGrid->particleCount(ijk)) {
				cell(ijk) = FLIPCellType::FLUID;
				particleGrid->iterateCell(ijk, [&](const size_t& id){
					if(particles[id]->type == ParticleType::SOLID) {
						cell(ijk) = FLIPCellType::SOLID;
						}
				});
			}
		}
	}

	void FLIP::storeParticleVelocities() {
		auto particles = particleGrid->getParticles();
		ponos::parallel_for(0, particles.size(), [this, &particles](size_t s, size_t e) {
			for(; s <= e; s++) {
				FLIPParticle *p = static_cast<FLIPParticle*>(particles[s]);
				p->v_copy = p->velocity;
				p->p_copy = p->position;
			}
				});
	}

	void FLIP::step() {
		//scene.generateParticles(curStep, particleGrid);
		scene.buildSolidLevelSet();

		// fixParticlesInSolids();
		storeParticleVelocities();
		particleGrid->computeDensity(density, maxDensity);
		applyForces();
		markCells();
		enforceBoundaries();
		project();
		enforceBoundaries();
		curStep++;
	}

	void FLIP::applyForces() {
		auto forces = scene.getForces(curStep);
		size_t forcesCount = forces.size();
		auto particles = particleGrid->getParticles();
		ponos::parallel_for(0, particles.size(), [=](size_t s, size_t e){
				for(; s <= e; s++)
					for(size_t i = 0; i < forcesCount; i++)
						particles[s]->velocity += forces[i] * dt;
		});
	}

	void FLIP::updateMACGrid() {
		size_t IN = curStep % 2;
		size_t OUT = (curStep + 1) % 2;
		float radius = dx * 2;
		ponos::parallel_for(0, dimensions[0] + 1, [this, IN, OUT, radius](size_t start, size_t end) {
				ponos::ivec3 m(start, 0, 0);
				ponos::ivec3 M(end + 1, dimensions[1] + 1, dimensions[2] + 1);
				ponos::ivec3 ijk;
				FOR_INDICES3D(m, dimensions, ijk) {
				if((ijk[0] >= dimensions[0] && ijk[1] >= dimensions[1]) ||
					(ijk[2] >= dimensions[2] && ijk[1] >= dimensions[1]) ||
					(ijk[2] >= dimensions[2] && ijk[0] >= dimensions[0])) continue;
					ponos::Point3 wpx = grid[IN]->getWorldPositionX(ijk[0], ijk[1], ijk[2]);
					ponos::Point3 wpy = grid[IN]->getWorldPositionY(ijk[0], ijk[1], ijk[2]);
					ponos::Point3 wpz = grid[IN]->getWorldPositionZ(ijk[0], ijk[1], ijk[2]);
					grid[OUT]->set(ijk[0], ijk[1], ijk[2], ponos::vec3(
							particleGrid->gather(ParticleAttribute::VELOCITY_X, wpx, radius),
							particleGrid->gather(ParticleAttribute::VELOCITY_Y, wpy, radius),
							particleGrid->gather(ParticleAttribute::VELOCITY_Z, wpz, radius)));
				}
				});
	}

	void FLIP::enforceBoundaries() {
		size_t IN = curStep % 2;
		size_t OUT = (curStep + 1) % 2;
		ponos::parallel_for(0, dimensions[0] + 1, [this, IN, OUT] (size_t s, size_t e) {
				ponos::ivec3 m(s, 0, 0);
				ponos::ivec3 M(e + 1, dimensions[1] + 1, dimensions[2] + 1);
				ponos::ivec3 ijk;
				FOR_INDICES3D(m, M, ijk) {
					if((ijk < dimensions) && cell(ijk) == FLIPCellType::SOLID) {
						grid[OUT]->setU(ijk[0], ijk[1], ijk[2], 0.f);
						grid[OUT]->setU(ijk[0] + 1, ijk[1], ijk[2], 0.f);
						grid[OUT]->setV(ijk[0], ijk[1], ijk[2], 0.f);
						grid[OUT]->setV(ijk[0], ijk[1] + 1, ijk[2], 0.f);
						grid[OUT]->setW(ijk[0], ijk[1], ijk[2], 0.f);
						grid[OUT]->setW(ijk[0], ijk[1], ijk[2] + 1, 0.f);
					}
					if(ijk[0] == dimensions[0])
						grid[OUT]->setU(ijk[0], ijk[1], ijk[2], 0.f);
					if(ijk[1] == dimensions[1])
						grid[OUT]->setV(ijk[0], ijk[1], ijk[2], 0.f);
					if(ijk[2] == dimensions[2])
						grid[OUT]->setW(ijk[0], ijk[1], ijk[2], 0.f);
				}
				});
	}

	void FLIP::project() {
		size_t IN = curStep % 2;
		grid[IN]->computeDivergence();

		using namespace openvdb;
		typedef math::pcg::SparseStencilMatrix<double, 7> MatrixType;
		typedef math::pcg::IncompleteCholeskyPreconditioner<MatrixType> CholeskyPrecond;

		const math::pcg::SizeType rows = dimensions[0] * dimensions[1] * dimensions[2];
		MatrixType A(rows);
		MatrixType::VectorType x(rows, 0.0), b(rows, 1.0);
		float pscale = dt / (dx *dx * density);
		float rscale = 1.f / dx;
		FOR_INDICES0_3D_ijk(dimensions, i, j, k) {
			int INDEX = i * j * k;
			if(cell(i, j, k) == FLIPCellType::FLUID) {
				b[INDEX] = grid[IN]->getDivergence(i, j, k);
				float v = pscale;
				if(cell(i - 1, j, k) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] -= rscale * (grid[IN]->u(i, j, k) - 0.f);
				if(cell(i + 1, j, k) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] += rscale * (grid[IN]->u(i + 1, j, k) - 0.f);
				if(cell(i, j - 1, k) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] -= rscale * (grid[IN]->v(i, j, k) - 0.f);
				if(cell(i, j + 1, k) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] += rscale * (grid[IN]->v(i, j + 1, k) - 0.f);
				if(cell(i, j, j - 1) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] -= rscale * (grid[IN]->w(i, j, k) - 0.f);
				if(cell(i, j, j + 1) != FLIPCellType::SOLID) v += pscale;
				else b[INDEX] += rscale * (grid[IN]->w(i, j, k + 1) - 0.f);
				A.setValue(INDEX, INDEX, v);
			}
		}
		CholeskyPrecond pA(A);
    // Solve A * x = b for x.
		openvdb::math::pcg::State result = math::pcg::solve(
        A, b, x, pA, openvdb::math::pcg::terminationDefaults<double>());

    ASSERT(result.success);
		// subtract pressure gradient
		float scale = dt / (density * dx);
		FOR_INDICES0_3D_ijk(dimensions, i, j, k) {
			int INDEX = i * j * k;
			if(cell(i, j, k) == FLIPCellType::FLUID) {
				grid[IN]->setU(i, j, k, grid[IN]->u(i, j, k) - scale * x[INDEX]);
				grid[IN]->setU(i + 1, j, k, grid[IN]->u(i + 1, j, k) + scale * x[INDEX]);
				grid[IN]->setV(i, j, k, grid[IN]->v(i, j, k) - scale * x[INDEX]);
				grid[IN]->setV(i, j + 1, k, grid[IN]->v(i, j + 1, k) + scale * x[INDEX]);
				grid[IN]->setW(i, j, k, grid[IN]->w(i, j, k) - scale * x[INDEX]);
				grid[IN]->setW(i, j, k + 1, grid[IN]->w(i, j, k + 1) + scale * x[INDEX]);
			}
		}
	}

	void FLIP::extrapolateVelocities() {
	}

} // poseidon namespace
