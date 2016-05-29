#pragma once

//#include <GL/gl.h>
//#include <GL/glu.ha>
#define GLFW_INCLUDE_GLU
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <ponos.h>
using ponos::Point2;
using ponos::vec2;
using ponos::vec3;

namespace aergia {

  void glVertex(Point2 v);
  void glVertex(vec2 v);

} // aergia namespace
