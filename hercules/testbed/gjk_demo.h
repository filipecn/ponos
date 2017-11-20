#include "demo.h"

#include <stdlib.h>
#include <time.h>

class GJKDemo : public Demo {
	public:
		GJKDemo(int w, int h)
			: Demo(w, h) { srand(time(NULL)); }
		void init() override {
			// set camera
			camera.setPos(ponos::vec2(0, 0));
			camera.resize(width, height);
			createPolygon(A);
			createPolygon(B);
			angleA = angleB = 0;
			vA = 0.0001 * ponos::vec2(rand() % 2 + 1, rand() % 2 + 1);
			vB = 0.0001 * ponos::vec2(rand() % 2 + 1, rand() % 2 + 1);
			lastTick = clock.tackTick();
		}
		void update(double dt) override {
			applyTransform(A, pA, vA, angleA, dt);
			applyTransform(B, pB, vB, angleB, dt);
		}
		void draw() override {
			aergia::GraphicsDisplay& gd = aergia::GraphicsDisplay::instance();
			gd.clearScreen(1.f, 1.f, 1.f, 0.f);
			glEnable (GL_BLEND);
			glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			camera.look();
			glLineWidth(1.f);
			if(hercules::cds::GJK::intersect(A, B))
				glLineWidth(5.f);
			glColor3f(1,0,0);
			aergia::draw_polygon(A);
			glColor3f(0,0,1);
			aergia::draw_polygon(B);
		}
	private:
		void createPolygon(ponos::Polygon &p) {
			float radius = static_cast<float>(rand() % width + 50) / static_cast<float>(width);
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
		void applyTransform(ponos::Polygon &polygon, ponos::Point2& pos, ponos::vec2 &v, float &angle, double dt) {
				pos += v * dt;
				if(pos.x > 1 || pos.x < -1)
					v.x *= -1;
				if(pos.y > 1 || pos.y < -1)
					v.y *= -1;
				ponos::Transform2D t;
				t.translate(v * dt);
				for(auto &p : polygon.vertices) {
					//	p = t(p);
					p.x += v.x * dt;
					p.y += v.y * dt;
				}
			}

		aergia::Camera2D camera;
		ponos::Polygon A, B;
		ponos::vec2 vA, vB;
		ponos::Point2 pA, pB;
		float angleA, angleB;
};
