#include "demo.h"

#include <stdlib.h>
#include <time.h>

class GJKTransformDemo : public Demo {
	public:
		GJKTransformDemo(int w, int h)
			: Demo(w, h) { srand(time(NULL)); }
		void init() override {
			// set camera
			camera.setPos(ponos::vec2(0, 0));
			camera.resize(width, height);
			createPolygon(sA);
			createPolygon(sB);
			lastTick = clock.tackTick();
      A = new hercules::_2d::RigidBody(nullptr, std::shared_ptr<ponos::Shape>(&sA));
      B = new hercules::_2d::RigidBody(nullptr, std::shared_ptr<ponos::Shape>(&sB));
      A->velocity = 0.0001 * ponos::vec2(1, 2);
      B->velocity = 0.0001 * ponos::vec2(3, -2);
      angleA = angleB = 1.f;
		}
		void update(double dt) override {
      update(A, dt, angleA);
      update(B, dt, angleB);
    }
		void draw() override {
			aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
			gd.clearScreen(1.f, 1.f, 1.f, 0.f);
			glEnable (GL_BLEND);
			glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			camera.look();
			glLineWidth(1.f);
			if(hercules::cds::GJK::intersect(
			*static_cast<ponos::Polygon*>(A->getShape()),
			*static_cast<ponos::Polygon*>(B->getShape()), &A->transform, &B->transform))
				glLineWidth(5.f);
			glColor3f(0,0,1);
			aergia::draw_polygon(*static_cast<ponos::Polygon*>(A->getShape()), &A->transform);
			glColor3f(1,0,1);
			aergia::draw_polygon(*static_cast<ponos::Polygon*>(B->getShape()), &B->transform);
		}
  private:
    void createPolygon(ponos::Polygon &p) {
      float radius = static_cast<float>(rand() % (width / 3) + 5) / static_cast<float>(width);
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
    void update(hercules::_2d::RigidBody* body, double dt, float& angle) {
      ponos::vec2 pos = body->transform.getTranslate();
      pos += body->velocity * dt;
      if(pos.x > 1 || pos.x < -1) {
        body->velocity.x *= -1;
        angle *= -1.f;
      }
      if(pos.y > 1 || pos.y < -1)
      body->velocity.y *= -1;
      body->transform.rotate(angle);
      body->transform.translate(body->velocity * dt);
    }

    aergia::Camera2D camera;
    hercules::_2d::RigidBody* A, *B;
    ponos::Polygon sA, sB;
    float angleA, angleB;
  };
