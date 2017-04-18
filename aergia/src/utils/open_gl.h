#ifndef AERGIA_UTILS_OPEN_GL_H
#define AERGIA_UTILS_OPEN_GL_H

#include <GL/glew.h>
#include <GL/freeglut.h>

//#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_GLU
//#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <ponos.h>

#include "colors/color.h"

#define GL_DEBUG

namespace aergia {

#ifdef GL_DEBUG
#define CHECK_GL_ERRORS printOglError(__FILE__, __LINE__)
#define CHECK_FRAMEBUFFER checkFramebuffer()
#else
#define CHECK_GL_ERRORS
#define CHECK_FRAMEBUFFER
#endif

bool initGLEW();
/* info
 * @shader **[in]** shader id
 * Print out the information log for a shader object
 */
void printShaderInfoLog(GLuint shader);
/* info
 * @program **[in]** program id
 * Print out the information log for a program object
 */
void printProgramInfoLog(GLuint program);
/* error check
 * @file **[in]** caller file
 * @line **[in]** caller line
 * @return **true** if any OpenGL error occured
 */
bool printOglError(const char *file, int line);
/* error check
 * Check framebuffer is COMPLETE
 * @return **false** if NO errors occured
 */
bool checkFramebuffer();
/* query
 * @major **[out]** receives major version
 * @minor **[out]** receives minor version
 * Retreives opengl version. Any error is sent to **stderr**.
 */
void getGlVersion(int *major, int *minor);
/* glVertex
 * Same as glVertex3f
 */
void glVertex(ponos::Point3 v);
/* glVertex
 * Same as glVertex2f
 */
void glVertex(ponos::Point2 v);
/* glVertex
 * Same as glVertex2f
 */
void glVertex(ponos::vec2 v);
/* glVertex
 * Same as glVertex2f
 */
void glVertex(ponos::Point<float, 2> v);

void glColor(Color c);

void glApplyTransform(const ponos::Transform &transform);

} // aergia namespace

#endif // AERGIA_UTILS_OPEN_GL_H
