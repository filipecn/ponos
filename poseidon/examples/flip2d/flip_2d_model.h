#include <aergia.h>
#include <ponos.h>

template<typename ParticleType>
class FLIP2DSceneModel : public aergia::SceneObject {
	public:
		FLIP2DSceneModel(const FLIP2D<ParticleType>* fl) {
			flip.reset(fl);
	//		transform = ponos::scale(flip->dx, flip->dx);
		}

		void draw() const override {
			glColor4f(0.f, 0.f, 1.f, 0.8f);
			for(auto mesh : flip->scene->getLiquidsGeometry())
				aergia::draw_mesh(mesh, &mesh->getTransform());
			glColor4f(0.3f, 0.4f, 0.2f, 0.8f);
			for(auto mesh : flip->scene->getStaticSolidsGeometry())
				aergia::draw_mesh(mesh, &mesh->getTransform());
		}

	private:
		std::shared_ptr<const FLIP2D<ParticleType> > flip;
		ponos::Transform2D transform;
};
