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
      shapes.resize(2000);
      for(int i = 0; i < 2000; i++) {
        createPolygon(shapes[i]);
        ponos::IndexPointer<hercules::_2d::RigidBody> ptr =
          objects.create(nullptr, &shapes[i]);
        ptr->velocity = 0.1 * ponos::vec2(.2f / float(rand() % 1000 + 1), .2f / float(rand() % 1000 + 1));
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
        aergia::draw_polygon(*static_cast<const ponos::Polygon*>((*it)->getShape()));
      for(size_t i = 0; i < contacts.size(); i++) {
        glLineWidth(5.f);
        const ponos::Polygon& A = *static_cast<const ponos::Polygon*>(static_cast<hercules::_2d::RigidBody*>(contacts[i].a)->getShape());
        const ponos::Polygon& B = *static_cast<const ponos::Polygon*>(static_cast<hercules::_2d::RigidBody*>(contacts[i].b)->getShape());
        aergia::draw_polygon(A);
        aergia::draw_polygon(B);
      }
    }
	private:
		void createPolygon(ponos::Polygon &p) {
			float radius = std::min(0.1f, static_cast<float>(rand() % (width / 2) + 10) / static_cast<float>(width));
      std::cout << radius << std::endl;
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
				for(auto &p : static_cast<ponos::Polygon*>(body.getShape())->vertices) {
					p.x += body.velocity.x * dt;
					p.y += body.velocity.y * dt;
				}
        body.setTransform(ponos::translate(pos));
			}

    std::vector<ponos::Polygon> shapes;
    hercules::cds::AABBSweep<hercules::_2d::RigidBody> objects;
		aergia::Camera2D camera;
};
