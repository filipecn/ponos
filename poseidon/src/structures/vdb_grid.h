#ifndef POSEIDON_STRUCTURES_VDB_GRID_H
#define POSEIDON_STRUCTURES_VDB_GRID_H

#include <ponos.h>

#include <openvdb/openvdb.h>

namespace poseidon {

	/* continuous grid
	 * Implements a continuous grid using OpenVDB library.
	 */
	class VDBGrid : public ponos::CGridInterface<float> {
		public:
			/* Constructor.
			 * @d **[in]** dimensions
			 * @b **[in]** background (default value)
			 * @s **[in]** scale
			 * @o **[in]** offset
			 */
			VDBGrid(const ponos::ivec3& d, const float& b, const float& s, const ponos::vec3& o);
			virtual ~VDBGrid() {}
			/* @inherit */
			void set(const ponos::ivec3& i, const float& v) override;
			/* @inherit */
			float operator()(const ponos::ivec3& i) const override;
			/* @inherit */
			float operator()(const int& i, const int&j, const int& k) const;
			/* @inherit */
			float operator()(const ponos::vec3& i) const override;
			/* @inherit */
			float operator()(const float& i, const float&j, const float& k) const override;

		private:
			using ValueType = openvdb::FloatGrid::ValueType;
			openvdb::FloatGrid::Ptr grid;
	};

} // poseidon namespace

#endif // POSEIDON_STRUCTURES_VDB_GRID_H

