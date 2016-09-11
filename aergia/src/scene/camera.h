#ifndef AERGIA_SCENE_CAMERA_H
#define AERGIA_SCENE_CAMERA_H

#include <ponos.h>

namespace aergia {

	class Camera {
		public:
			Camera();

			friend class CameraModel;

			void look();
			void resize(float w, float h);
			void setZoom(float z);
			ponos::Point3 getPos() const { return pos; }
			void setPos(ponos::Point3 p);
			ponos::Point3 getTarget() const { return target; }
			void setTarget(ponos::Point3 p);
			void setFov(float f);
			void setFar(float f);
			void setNear(float n);
			void update();
			ponos::Transform getTransform() const;
			ponos::Point3 viewPointOnWorldCoord() const;
			// p must be in norm dev coord (windowCoordToNormDevCoord)
			ponos::Line viewLineFromWindow(ponos::Point2 p) const;
			ponos::Ray3 pickRay(ponos::Point2 p) const;
			ponos::Plane viewPlane(ponos::Point3 p) const;
		//private:
			float ratio;
			float zoom;
			float near, far;
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
