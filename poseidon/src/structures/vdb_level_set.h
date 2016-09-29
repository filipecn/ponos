#ifndef POSEIDON_STRUCTURES_LEVEL_SET_H
#define POSEIDON_STRUCTURES_LEVEL_SET_H

#include "elements/particle.h"

#include <aergia.h>
#include <ponos.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>
#include <vector>

namespace poseidon {
/*
	class ParticleList {
		public:
			ParticleList(){ }

			ParticleList(const std::vector<Particle*> &plist)
			: particles(plist) {}

			~ParticleList(){ }

			int size() const {
				return particles.size();
			}

			void getPos(size_t n, openvdb::Vec3R& pos) const {
				pos = openvdb::Vec3f(
						particles[n]->position.x,
						particles[n]->position.y,
						particles[n]->position.z);
			}

			void getPosRad(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad) const {
				pos = openvdb::Vec3f(
						particles[n]->position.x,
						particles[n]->position.y,
						particles[n]->position.z);
				rad = particles[n]->density;
				rad = .5f;
				if(particles[n]->invalid){
					rad = 0.0f;
				}
			}

			void getPosRadVel(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad,
					openvdb::Vec3R& vel) const {
				pos = openvdb::Vec3f(
						particles[n]->position.x,
						particles[n]->position.y,
						particles[n]->position.z);
				vel = openvdb::Vec3f(
						particles[n]->velocity.x,
						particles[n]->velocity.y,
						particles[n]->velocity.z);
				rad = particles[n]->density;
				rad = .5f;
				if(particles[n]->invalid) {
					rad = 0.0f;
				}
				void getAtt(size_t n, openvdb::Index32& att) const { att = n; }

				private:
				const std::vector<Particle*> &particles;
			};
*/
	class VDBLevelSet : public ponos::CGridInterface<float> {
  	public:
			/* Constructor
			 * @m **[in]** mesh
			 * create a level set from a mesh
			 */
			VDBLevelSet(const aergia::RawMesh *m, ponos::Transform t);
			virtual ~VDBLevelSet() {}
			/* @inherit */
			void set(const ponos::ivec3& ijk, const float& v) override;
			/* @inherit */
			float operator()(const ponos::ivec3& ijk) const override;
			/* @inherit */
			float operator()(const int& i, const int& j, const int& k) const;
			/* @inherit */
			float  operator()(const ponos::vec3& xyz) const override;
			/* @inherit */
			float operator()(const float& x, const float &y, const float& z) const override;
			/* merge
			 * @ls **[out]**
			 * @return
			 */
			void merge(const VDBLevelSet *ls);
			void copy(const VDBLevelSet *ls);

			const openvdb::FloatGrid::Ptr& getVDBGrid() const;

		private:
			openvdb::FloatGrid::Ptr grid;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_LEVEL_SET_H

