#ifndef AERGIA_SCENE_SCENE_OBJECT_H
#define AERGIA_SCENE_SCENE_OBJECT_H

#include <ponos.h>

namespace aergia {

	class SceneObject {
  	public:
	 		SceneObject() {}
			virtual ~SceneObject() {}

			virtual void draw() = 0;

			ponos::Transform t;
	};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
