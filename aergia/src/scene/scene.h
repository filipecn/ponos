#ifndef AERGIA_SCENE_SCENE_H
#define AERGIA_SCENE_SCENE_H

#include "scene/scene_object.h"

#include <ponos.h>

#include <memory>

namespace aergia {

	/* scene
	 * The scene stores the list of objects to be rendered.
	 * It is possible how these objects are arranged in memory by setting
	 * a **StructureType**. The default organization is a flat array with no acceleration schemes.
	 */
	template<template <typename> class StructureType = ponos::Array>
		class Scene {
			public:
				Scene() {}
				virtual ~Scene() {}

				void add(SceneObject* o) {
					s.add(std::shared_ptr<SceneObject>(o));
				}

				void render() {
					s.iterate([](const std::shared_ptr<SceneObject>& o) { o->draw(); });
				}

			private:
				StructureType<std::shared_ptr<SceneObject> > s;
		};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_H

