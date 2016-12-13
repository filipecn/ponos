#include "flip_2d.h"

FLIP2DScene::FLIP2DScene(const char *filename) {
	load(filename);
}

FLIP2DScene::~FLIP2DScene() {}

void FLIP2DScene::load(const char *filename) {
	FILE *fp = fopen(filename, "r");
	if(!fp)
		return;
	fclose(fp);
}

void FLIP2DScene::addForce(const ponos::vec2 &f) {
	forces.emplace_back(f);
}

void FLIP2DScene::generateParticles(int frame, ZParticleGrid2D<FLIPParticle2D> *pg) {
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

const std::vector<ponos::Mesh2D*>& FLIP2DScene::getLiquidsGeometry() const {
	return liquidsGeometry;
}

const std::vector<ponos::Mesh2D*>& FLIP2DScene::getStaticSolidsGeometry() const {
	return staticSolidsGeometry;
}

const std::vector<ponos::Mesh2D*>& FLIP2DScene::getSolidsGeometry() const {
	return solidsGeometry;
}

const std::vector<FLIPParticle2D*>& FLIP2DScene::getLiquidParticles() const {
	return liquidParticles;
}

void FLIP2DScene::buildStaticSolidLevelSet() {
	for(size_t i = 0; i < staticSolidsGeometry.size(); i++) {
		if(i == 0)
			staticSolidLevelSet.reset(new ponos::LevelSet2D(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
		else staticSolidLevelSet->merge(new ponos::LevelSet2D(staticSolidsGeometry[i]->getMesh(), staticSolidsGeometry[i]->getTransform()));
	}
}

void FLIP2DScene::buildSolidLevelSet() {
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

bool FLIP2DScene::isPointInsideSolid(const ponos::Point2 &p, int f, size_t *id) const {
	for(size_t i = 0; i < solidsGeometry.size(); i++) {
		if(solidsGeometry[i]->intersect(p)) {
			*id = i;
			return true;
		}
	}
	return false;
}

void FLIP2D::setup() {}

bool FLIP2D::init() { return true; }

void FLIP2D::step() {}
