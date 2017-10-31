#include "demo.h"

#include <stdlib.h>
#include <time.h>

class AABBSweepDemo : public Demo {
	public:
		AABBSweepDemo(int w, int h)
			: Demo(w, h) { srand(time(NULL)); }
		void init() override {
			// set camera
			camera.setPos(ponos::vec2(0, 0));
			camera.resize(width, height);
      shapes.resize(5);
      for(int i = 0; i < 5; i++) {
       	shapes[i].reset(new ponos::Polygon());
				createPolygon(*static_cast<ponos::Polygon*>(shapes[i].get()));
			}
      for (int i = 0; i < 50; i++) {
        objects.create(nullptr, shapes[i % 5]);
        auto ptr = objects.get(i);
        ptr->userData = new int(i);
      }
      for(hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(objects); it.next(); ++it) {
        (*it)->velocity = 0.4 * ponos::vec2(.2f / float(rand() % 1000 + 1),
                                            .2f / float(rand() % 1000 + 1));
        (*it)->transform.translate(ponos::vec2(
          (static_cast<float>(rand() % width) / (1.f * width)) * 2.f - 1.f,
          (static_cast<float>(rand() % height) / (1.f * height)) * 2.f - 1.f));
      }
			lastTick = clock.tackTick();
		}
    void update(double dt) override {
      for(hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(objects); it.next(); ++it)
        applyTransform(**it, dt);
    }
    void draw() override {
      aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
      gd.clearScreen(1.f, 1.f, 1.f, 0.f);
      glEnable (GL_BLEND);
      glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      camera.look();
      std::vector<hercules::cds::Contact2D> contacts = objects.collide();
      glColor3f(0,0,0);
      glLineWidth(1.f);
      for(hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(objects); it.next(); ++it)
        aergia::draw_polygon(*static_cast<const ponos::Polygon*>((*it)->getShape()), &(*it)->transform);
      for(size_t i = 0; i < contacts.size(); i++) {
        glLineWidth(5.f);
        hercules::_2d::RigidBody* bA = static_cast<hercules::_2d::RigidBody*>(contacts[i].a);
        hercules::_2d::RigidBody* bB = static_cast<hercules::_2d::RigidBody*>(contacts[i].b);
        const ponos::Polygon& A = *static_cast<const ponos::Polygon*>(bA->getShape());
        const ponos::Polygon& B = *static_cast<const ponos::Polygon*>(bB->getShape());
        aergia::draw_polygon(A, &bA->transform);
        aergia::draw_polygon(B, &bB->transform);
        bA->toDelete = true;
        bB->toDelete = true;
        //std::cout << "collision " <<
        //  *static_cast<int*>(bA->userData) <<
        //  " " <<
        //  *static_cast<int*>(bB->userData) << std::endl;
      }
      objects.cleanDeleted();
    }
	private:
		void createPolygon(ponos::Polygon &p) {
			float radius = std::min(0.1f, static_cast<float>(rand() % (width / 2) + 10) / static_cast<float>(width));
			int angle = 360;
			p.vertices.emplace_back(radius * cos(TO_RADIANS(0)), radius * sin(TO_RADIANS(0)));
			while(1) {
				angle -= rand() % (angle / 2) + 12;
				if(angle <= 1 || p.vertices.size() > 5)
					break;
				float a = static_cast<float>(angle);
				p.vertices.emplace_back(radius * cos(TO_RADIANS(a)), radius * sin(TO_RADIANS(a)));
			}
		}
		void applyTransform(hercules::_2d::RigidBody& body, double dt) {
        ponos::vec2 pos = body.transform.getTranslate();
        pos += body.velocity * dt;
				if(pos.x > 1 || pos.x < -1)
					body.velocity.x *= -1;
				if(pos.y > 1 || pos.y < -1)
					body.velocity.y *= -1;
        body.transform.rotate(1.f);
        body.transform.translate(body.velocity * dt);
        body.setTransform(body.transform);
      }

    std::vector<std::shared_ptr<ponos::Shape> > shapes;
    hercules::cds::AABBSweep<hercules::_2d::RigidBody> objects;
		aergia::Camera2D camera;
};
