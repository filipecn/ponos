#ifndef AERGIA_SCENE_SCENE_H
#define AERGIA_SCENE_SCENE_H

#include "scene/scene_object.h"
#include "utils/open_gl.h"

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
					s.add(o);
				}

				void render() {
					float pm[16];
					transform.matrix().column_major(pm);
					glMultMatrixf(pm);
					s.iterate([](const SceneObject *o) { o->draw(); });
				}

				SceneObject* intersect(const ponos::Ray3& ray, float *t = nullptr) {
					ponos::Transform tr = ponos::inverse(transform);
					ponos::Point3 ta = tr(ray.o);
					ponos::Point3 tb = tr(ray(1.f));
					return s.intersect(ponos::Ray3(ta, tb - ta), t);
				}

				ponos::Transform transform;

			private:
				StructureType<SceneObject> s;
		};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_H

