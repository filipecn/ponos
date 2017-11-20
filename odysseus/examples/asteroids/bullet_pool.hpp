#include <odysseus.h>

using namespace odysseus;

#include <cstring>
#include <memory>
#include <vector>

#include <ponos.h>

class Bullet : public GameObject {
	public:
		Bullet() {
			state_.next = nullptr;
		}
		void set(std::shared_ptr<GraphicsComponent> graphics,
				std::shared_ptr<PhysicsComponent> physics) {
			graphics_ = graphics;
			physics_ = physics;
		}
		void setNext(Bullet* n) {
			state_.next = n;
		}
		Bullet* getNext() {
			return state_.next;
		}
		void init(ponos::Point2 pos, ponos::vec2 vel) {
			state_.bullet.vel = vel;
		}

		virtual void render() override {}
		virtual void processInput() override {}
		virtual void update() override {}

	private:
		union State {
			struct {
			  ponos::vec2 vel;
			} bullet;
			Bullet* next;
			State() { memset(this, 0, sizeof(State)); }
		} state_;
};

class BulletGraphics : public GraphicsComponent {
	public:
		BulletGraphics() {}
		virtual void render(GameObject* bullet) override {
			ponos::Transform2D t = bullet->transform * transform;
			glColor3f(0,0,1);
			glBegin(GL_LINE_LOOP);
				aergia::glVertex(t(vertices[0]));
				aergia::glVertex(t(vertices[1]));
				aergia::glVertex(t(vertices[2]));
			glEnd();
		}
};

class BulletPhysics : public PhysicsComponent {
	public:
		BulletPhysics() {}
		virtual void update(std::shared_ptr<GameObject> obj, hercules::World &world) override {}
};

class BulletManager {
	public:
		BulletManager(int size);

		void create(ponos::Point2 pos, ponos::vec2 vel);
		void destroy(Bullet* bullet);
		void shoot(ponos::Transform2D t, hercules::World &world);
	private:
		Bullet* newBullet;
		std::vector<Bullet> pool;
		std::shared_ptr<GraphicsComponent> graphics;
		std::shared_ptr<PhysicsComponent> physics;
		std::shared_ptr<hercules::Fixture> bulletFixture;
};

BulletManager::BulletManager(int size) {
	pool.resize(size);
	newBullet = &pool[0];
	for (unsigned int i = 0; i < pool.size() - 1; i++) {
		pool[i].setNext(&pool[i+1]);
	}
	graphics.reset(new BulletGraphics());
	physics.reset(new BulletPhysics());
	for (unsigned int i = 0; i < pool.size(); i++) {
		pool[i].set(graphics, physics);
	}
	bulletFixture.reset(new hercules::Fixture());
	hercules::Polygon *polygon = new hercules::Polygon();
	polygon->vertices.emplace_back(0.0f, 0.0f);
	polygon->vertices.emplace_back(0.1f, 0.0f);
	polygon->vertices.emplace_back(0.1f, 0.3f);
	polygon->vertices.emplace_back(0.0f, 0.3f);
	bulletFixture->shape.reset(polygon);
}

void BulletManager::create(ponos::Point2 pos, ponos::vec2 vel) {
	if(newBullet == nullptr)
		return;
	Bullet* bullet = newBullet;
	newBullet = newBullet->getNext();
	bullet->init(pos, vel);
}

void BulletManager::destroy(Bullet* bullet) {
	if(bullet == nullptr)
		return;
	bullet->setNext(newBullet);
	newBullet = bullet;
}

void BulletManager::shoot(ponos::Transform2D t, hercules::World &world) {
	// ponos::Point2 pos = t.getTranslate();
	// ponos::vec2 vel = t(ponos::Point2(1.f, 0.f));
	// create(pos, vel);
}
