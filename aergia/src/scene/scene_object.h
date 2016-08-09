#pragma once

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

