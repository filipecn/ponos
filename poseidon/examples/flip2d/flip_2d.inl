
template<typename ParticleType>
FLIP2DScene<ParticleType>::FLIP2DScene(const char *filename) {
	load(filename);
}

template<typename ParticleType>
FLIP2DScene<ParticleType>::~FLIP2DScene() {}

template<typename ParticleType>
void FLIP2DScene<ParticleType>::load(const char *filename) {
	FILE *fp = fopen(filename, "r");
	if(!fp)
		return;
	char line[1000];
	while(fgets(line, 1000, fp) != nullptr) {
		char command = line[0];
		fgets(line, 1000, fp);
		int lx, ly, tx, ty;
		switch(command) {
			case 'l': {
				sscanf(line, "%d %d %d %d", &lx, &ly, &tx, &ty);
				ponos::RawMesh *m = new ponos::RawMesh();
				m->vertices.emplace_back(lx); m->vertices.emplace_back(ly);
				m->vertices.emplace_back(tx); m->vertices.emplace_back(ly);
				m->vertices.emplace_back(tx); m->vertices.emplace_back(ty);
				m->vertices.emplace_back(lx); m->vertices.emplace_back(ty);
				m->elementSize = 2;
				m->elementCount = 4;
				m->indices.resize(8);
				for(int i = 0; i < 4; i++) {
					m->indices[(i * 2) % 4].vertexIndex = i;
					m->indices[(i * 2 + 1) % 4].vertexIndex = (i + 1) % 4;
				}
				m->computeBBox();
				liquidsGeometry.emplace_back(new ponos::Mesh2D(m, ponos::Transform2D()));
								} break;
			case 's':
				sscanf(line, "%d %d %d %d", &lx, &ly, &tx, &ty);
				break;
		}
	}
	fclose(fp);
}

template<typename ParticleType>
void FLIP2DScene<ParticleType>::addForce(const ponos::vec2 &f) {
	forces.emplace_back(f);
}

template<typename ParticleType>
void FLIP2DScene<ParticleType>::generateParticles(int frame, ZParticleGrid2D<ParticleType> *pg) {
	/*/ generate liquid particles
	for(size_t l = 0; l < liquidsGeometry.size(); l++) {
		ponos::BBox2D bbox = liquidsGeometry[l]->getBBox();
		ponos::ivec2 lmin = ponos::floor(ponos::vec2(pg->toGrid(bbox.pMin)));
		ponos::ivec2 lmax = ponos::ceil(ponos::vec2(pg->toGrid(bbox.pMax)));
		lmin = ponos::max(lmin, ponos::ivec2());
		lmax = ponos::min(lmax, pg->dimensions + ponos::ivec2(1.0f));
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
	pg->update();
*/}

template<typename ParticleType>
const std::vector<ponos::Mesh2D*>& FLIP2DScene<ParticleType>::getLiquidsGeometry() const {
	return liquidsGeometry;
}

template<typename ParticleType>
const std::vector<ponos::Mesh2D*>& FLIP2DScene<ParticleType>::getStaticSolidsGeometry() const {
	return staticSolidsGeometry;
}

template<typename ParticleType>
const std::vector<ponos::Mesh2D*>& FLIP2DScene<ParticleType>::getSolidsGeometry() const {
	return solidsGeometry;
}

template<typename ParticleType>
const std::vector<ParticleType*>& FLIP2DScene<ParticleType>::getLiquidParticles() const {
	return liquidParticles;
}

template<typename ParticleType>
void FLIP2DScene<ParticleType>::buildStaticSolidLevelSet() {
	for(size_t i = 0; i < staticSolidsGeometry.size(); i++) {
		if(i == 0)
			staticSolidLevelSet.reset(new ponos::LevelSet2D(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
		else staticSolidLevelSet->merge(new ponos::LevelSet2D(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
	}
}

template<typename ParticleType>
void FLIP2DScene<ParticleType>::buildSolidLevelSet() {
	for(size_t i = 0; i < solidsGeometry.size(); i++) {
		if(i == 0)
			staticSolidLevelSet.reset(new ponos::LevelSet2D(solidsGeometry[i]->getMesh(), solidsGeometry[i]->getTransform()));
		else staticSolidLevelSet->merge(new ponos::LevelSet2D(solidsGeometry[i]->getMesh(), solidsGeometry[i]->getTransform()));
	}
	if(solidsGeometry.size())
		solidLevelSet->copy(staticSolidLevelSet.get());
	else
		solidLevelSet->merge(staticSolidLevelSet.get());
}

template<typename ParticleType>
bool FLIP2DScene<ParticleType>::isPointInsideSolid(const ponos::Point2 &p, int f, size_t *id) const {
	for(size_t i = 0; i < solidsGeometry.size(); i++) {
		if(solidsGeometry[i]->intersect(p)) {
			*id = i;
			return true;
		}
	}
	return false;
}

template<typename ParticleType>
FLIP2D<ParticleType>::FLIP2D(FLIP2DScene<ParticleType> *s) {

}

template<typename ParticleType>
void FLIP2D<ParticleType>::setup() {
	particleGrid.reset(new ZParticleGrid2D<ParticleType>(dimensions[0], dimensions[1], ponos::BBox2D(ponos::Point2(), ponos::Point2(dimensions[0] * dx, dimensions[1] * dx))));
	for(int i = 0; i < 2; i++)
		grid[i] = new MacGrid2D<ponos::ZGrid>(dimensions[0], dimensions[1], dx);
}

template<typename ParticleType>
bool FLIP2D<ParticleType>::init() {
	ZParticleGrid2D<ParticleType> pg(10, 10, ponos::BBox2D(ponos::Point2(), ponos::Point2(10 * dx, 10 * dx)));
	ponos::ivec2 ij;
	FOR_INDICES0_2D(ponos::ivec2(10, 10), ij) {
		ponos::Point2 cp((ij[0] + 0.5f) * dx, (ij[1] + 0.5f) * dx);
		pg.addParticle(cp + ponos::vec2(-0.25f, -0.25f) * dx);
		pg.addParticle(cp + ponos::vec2( 0.25f, -0.25f) * dx);
		pg.addParticle(cp + ponos::vec2( 0.25f,  0.25f) * dx);
		pg.addParticle(cp + ponos::vec2( 0.25f,  0.25f) * dx);
	}
	maxDensity = 1.f;
	scene.generateParticles(0, particleGrid);
	return true;
}

template<typename ParticleType>
void FLIP2D<ParticleType>::step() {}
