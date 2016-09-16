#ifndef POSEIDON_STRUCTURES_LEVEL_SET_H
#define POSEIDON_STRUCTURES_LEVEL_SET_H

#include <ponos.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>

namespace poseidon {

	class VDBLevelSet : public ponos::CGridInterface<float> {
  	public:
	 		VDBLevelSet();
			virtual ~VDBLevelSet() {}
			/* @inherit */
			void set(const ivec3& i, const float& v) override;
			/* @inherit */
			float operator()(const ponos::ivec3& i) const override;
			/* @inherit */
			float operator()(const uint& i, const uint&j, const uint& k) const override;
			/* @inherit */
			float  operator()(const vec3& i) const override;
			/* @inherit */
			float operator()(const float& i, const float&j, const float& k) const override;

		private:
			openvdb::FloatGrid::Ptr grid;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_LEVEL_SET_H

