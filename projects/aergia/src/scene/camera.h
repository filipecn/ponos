#pragma once

#include <ponos.h>

namespace aergia {
	class Camera {
		public:
			Camera();

			void look();
			void resize(float w, float h);
			void setZoom(float z);
			void setPos(ponos::Point3 p);
			void setTarget(ponos::Point3 p);
			void update();
			ponos::Transform getTransform() const;
			ponos::Point3 viewPointOnWorldCoord() const;
			ponos::Line viewLineFromWindow(ponos::Point2 p) const;
		private:
			float ratio;
			float zoom;
			ponos::Point3 pos, target;
			ponos::vec3 up;
			ponos::vec2 display;
			ponos::vec2 clipSize;
			ponos::Transform projection;
			ponos::Transform view;
			ponos::Transform model;
	};
} // aergia namespace
