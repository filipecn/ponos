#ifndef HERCULES_CDS_COLLIDABLE_H
#define HERCULES_CDS_COLLIDABLE_H

namespace hercules {

	namespace cds {

		/* interface
		 * The **CDS**(Collision Detection System) only computes collisions between _collidable_
		 * objects (Objects must implement this class).
		 */
		class Collidable {
			public:
				Collidable() :
          colliding(false) {}
				virtual ~Collidable() {}

        bool colliding;
		};

	} // cds namespace

} // hercules namespace

#endif // HERCULES_CDS_COLLIDABLE_INTERFACE_H
