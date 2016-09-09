#ifndef PONOS_SPATIAL_ARRAY_H
#define PONOS_SPATIAL_ARRAY_H

#include "spatial/spatial_structure_interface.h"

#include <vector>

namespace ponos {

	template<typename ObjectType>
		class Array : public SpatialStructureInterface<ObjectType> {
			public:
				Array() {}
				virtual ~Array() {}
				/* @inherit */
				void add(const ObjectType& o) override {
					objects.push_back(o);
				}
				/* @inherit */
			  void iterate(std::function<void(const ObjectType& o)> f) override {
					for(const auto& e : objects)
						f(e);
				}

			private:
				std::vector<ObjectType> objects;
		};

} // ponos namespace

#endif // PONOS_SPATIAL_ARRAY_H

