#ifndef AERGIA_SCENE_CAMERA_H
#define AERGIA_SCENE_CAMERA_H

#include <ponos.h>

namespace aergia {

  class CameraInterface {
  public:
    CameraInterface() {}
    virtual ~CameraInterface() {}

    virtual ponos::Ray3 pickRay(ponos::Point2 p) const = 0;
    virtual void look() = 0;
		virtual void resize(float w, float h) = 0;
		virtual ponos::Transform getTransform() const = 0;
  };

	class Camera : public CameraInterface {
		public:
			Camera();

			friend class CameraModel;

			void look() override;
			void resize(float w, float h);
			void setZoom(float z);
			ponos::Point3 getPos() const { return pos; }
			void setPos(ponos::Point3 p);
			ponos::Point3 getTarget() const { return target; }
			void setTarget(ponos::Point3 p);
			void setFov(float f);
			void setFar(float f);
			void setNear(float n);
      float getNear() const { return near_; }
      float getFar() const { return far_; }
			void update();
			ponos::Transform getTransform() const override;
			ponos::Point3 viewPointOnWorldCoord() const;
			// p must be in norm dev coord (windowCoordToNormDevCoord)
			ponos::Line viewLineFromWindow(ponos::Point2 p) const;
			ponos::Ray3 pickRay(ponos::Point2 p) const override;
			ponos::Plane viewPlane(ponos::Point3 p) const;
		private:
			float ratio;
			float zoom;
			float near_, far_;
			float fov;
			ponos::Point3 pos, target;
			ponos::vec3 up;
			ponos::vec2 display;
			ponos::vec2 clipSize;
			ponos::Transform projection;
			ponos::Transform view;
			ponos::Transform model;
			ponos::Frustum frustum;
	};

} // aergia namespace

#endif // AERGIA_SCENE_CAMERA_H
