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
          wBBoxUpdated = false;
				}
				virtual ~RigidBody() {}

				void setFixture(Fixture *f) {
					fixture.reset(f);
				}
				void setShape(ponos::Shape *s) {
					shape.reset(s);
				}
				ponos::BBox2D getWBBox() override {
          if(!wBBoxUpdated) {
            wBBox = ponos::compute_bbox(*static_cast<ponos::Polygon*>(shape.get()));
            wBBoxUpdated = true;
          }
					return wBBox;
				}
        ponos::Shape* getShape() {
          return shape.get();
        }
        void setTransform(const ponos::Transform2D& t) {
          transform = t;
          wBBoxUpdated = false;
        }
        void destroy() {}
        ponos::Transform2D transform;
        ponos::vec2 velocity;

			private:
        bool wBBoxUpdated;
        ponos::BBox2D wBBox;
				std::shared_ptr<Fixture> fixture;
				std::shared_ptr<ponos::Shape> shape;
		};

	} // 2d namespace

} // hercules namespace

#endif // HERCULES_2D_RIGID_BODY_H
