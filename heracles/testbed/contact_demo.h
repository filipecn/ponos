#include "demo.h"

#include <stdlib.h>
#include <time.h>

class ContactDemo : public Demo {
public:
  ContactDemo(int w, int h) : Demo(w, h) { srand(time(NULL)); }
  void init() override {
    // set camera
    camera.setPos(ponos::vec2(0, 0));
    camera.resize(width, height);
    shapes.resize(5);
    for (int i = 0; i < 5; i++) {
      shapes[i].reset(new ponos::Circle(
          ponos::Point2(),
          std::min(0.1f,
                   static_cast<float>(rand() % (width / 2) + 10) /
                       static_cast<float>(width))));
    }
    for (int i = 0; i < 10; i++) {
      objects.create(nullptr, shapes[i % 5]);
      auto ptr = objects.get(i);
      ptr->userData = new int(i);
    }
    for (hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(
             objects);
         it.next(); ++it) {
      (*it)->velocity = 0.4 * ponos::vec2(.2f / float(rand() % 1000 + 1),
                                          .2f / float(rand() % 1000 + 1));
      (*it)->transform.translate(ponos::vec2(
          (static_cast<float>(rand() % width) / (1.f * width)) * 2.f - 1.f,
          (static_cast<float>(rand() % height) / (1.f * height)) * 2.f - 1.f));
    }
    lastTick = clock.tackTick();
  }
  void update(double dt) override {
    for (hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(
             objects);
         it.next(); ++it)
      applyTransform(**it, dt);
  }
  void draw() override {
    aergia::GraphicsDisplay &gd = aergia::GraphicsDisplay::instance();
    gd.clearScreen(1.f, 1.f, 1.f, 0.f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    camera.look();
    std::vector<hercules::cds::Contact2D> contacts = objects.collide();
    glColor4f(0, 0, 0, 0.1);
    for (hercules::cds::AABBSweep<hercules::_2d::RigidBody>::iterator it(
             objects);
         it.next(); ++it)
      aergia::draw_circle(
          *static_cast<const ponos::Circle *>((*it)->getShape()),
          &(*it)->transform);
    for (size_t i = 0; i < contacts.size(); i++) {
      glColor4f(0, 0, 0, 0.3);
      hercules::_2d::RigidBody *bA =
          static_cast<hercules::_2d::RigidBody *>(contacts[i].a);
      hercules::_2d::RigidBody *bB =
          static_cast<hercules::_2d::RigidBody *>(contacts[i].b);
      const ponos::Circle &A =
          *static_cast<const ponos::Circle *>(bA->getShape());
      const ponos::Circle &B =
          *static_cast<const ponos::Circle *>(bB->getShape());
      aergia::draw_circle(A, &bA->transform);
      aergia::draw_circle(B, &bB->transform);
      bA->toDelete = true;
      bB->toDelete = true;
      glPointSize(5.f);
      glColor3f(1, 0, 0);
      glBegin(GL_POINTS);
      aergia::glVertex(contacts[i].points[0].position);
      glEnd();
      glColor3f(0, 0, 1);
      glBegin(GL_LINES);
      aergia::glVertex(contacts[i].points[0].position);
      aergia::glVertex(contacts[i].points[0].position +
                       contacts[i].points[0].penetration *
                           ponos::vec2(contacts[i].normal));
      glEnd();
      // std::cout << "collision " << *static_cast<int*>(bA->userData) << " " <<
      // *static_cast<int*>(bB->userData) << std::endl;
    }
    // objects.cleanDeleted();
  }

private:
  void applyTransform(hercules::_2d::RigidBody &body, double dt) {
    ponos::vec2 pos = body.transform.getTranslate();
    pos += body.velocity * dt;
    if (pos.x > 1 || pos.x < -1)
      body.velocity.x *= -1;
    if (pos.y > 1 || pos.y < -1)
      body.velocity.y *= -1;
    body.transform.translate(body.velocity * dt);
    body.setTransform(body.transform);
  }

  std::vector<std::shared_ptr<ponos::Shape>> shapes;
  hercules::cds::AABBSweep<hercules::_2d::RigidBody> objects;
  aergia::Camera2D camera;
};
