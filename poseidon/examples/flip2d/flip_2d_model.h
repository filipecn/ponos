#include <aergia.h>
#include <ponos.h>

template<typename ParticleType>
class FLIP2DSceneModel : public aergia::SceneObject {
	public:
		FLIP2DSceneModel(const FLIP2DScene<ParticleType>* s) {
			scene.reset(s);
		}

		void draw() const override {
			for(auto mesh : scene->getLiquidsGeometry())
				aergia::draw_mesh(mesh);
			for(auto mesh : scene->getSolidsGeometry())
				aergia::draw_mesh(mesh);
		}

	private:
		std::shared_ptr<const FLIP2DScene<ParticleType> > scene;
};
