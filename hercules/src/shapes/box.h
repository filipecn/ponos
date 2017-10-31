#ifndef HERCULES_SHAPES_BOX_H
#define HERCULES_SHAPES_BOX_H

#include "rigid_body.h"

namespace hercules {

	/* Box rigid body.
	 * A rigid body with a box shape.
	 */
	class Box : public RigidBody {
  	public:
			/* Constructor.
			 * @w width
			 * @h height
			 * @d depth
			 *
			 * Creates a box with dimensions w * h * d.
			 */
	 		Box(float w, float h, float d);
			virtual ~Box() {}
	};

} // hercules namespace

#endif // HERCULES_SHAPES_BOX_H

