#ifndef HERCULES_2D_RIGID_BODY_H
#define HERCULES_2D_RIGID_BODY_H

#include "cds/cds.h"
#include "2d/fixture.h"

#include <ponos.h>

#include <memory>

namespace hercules {

	namespace _2d {

		class RigidBody : public cds::Collidable, public cds::AABBObjectInterface<ponos::BBox2D> {
			public:
				RigidBody() {}
				RigidBody(Fixture *f, ponos::Shape *s) {
					fixture.reset(f);
					shape.reset(s);
				}
				virtual ~RigidBody() {}

				void setFixture(Fixture *f) {
					fixture.reset(f);
				}
				void setShape(ponos::Shape *s) {
					shape.reset(s);
				}
				ponos::BBox2D getWBBox() override {
					return ponos::BBox2D();
				}
			private:
				std::shared_ptr<Fixture> fixture;
				std::shared_ptr<ponos::Shape> shape;
		};

	} // 2d namespace

} // hercules namespace

#endif // HERCULES_2D_RIGID_BODY_H

