#ifndef AERGIA_SCENE_SCENE_OBJECT_H
#define AERGIA_SCENE_SCENE_OBJECT_H

#include <ponos.h>

namespace aergia {

	/* interface
	 * The Scene Object Interface represents an drawable object that can be intersected
	 * by a ray.
	 */
	class SceneObject {
  	public:
	 		SceneObject() {}
			virtual ~SceneObject() {}

			/* draw
			 * render method
			 */
			virtual void draw() const = 0;
			/* query
			 * @r **[in]** ray
			 * @t **[out]** receives the parametric value of the intersection
			 * @return **true** if intersection is found
			 */
			virtual bool intersect(const ponos::Ray3 &r, float *t = nullptr) const { return false; }

			ponos::Transform t;
			bool selected;
	};

} // aergia namespace

#endif // AERGIA_SCENE_SCENE_OBJECT_H
