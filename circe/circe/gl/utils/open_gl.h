#ifndef CIRCE_UTILS_OPEN_GL_H
#define CIRCE_UTILS_OPEN_GL_H

#define GL_DEBUG
//#define GLEW_BUILD
//#define GLEW_STATIC
//#include <GL/glew.h>
#if !defined(GLAD_ALREADY_INCLUDED)
#include <glad/glad.h>
#endif // GLAD_ALREADY_INCLUDED

// #include <nanogui/nanogui.h>
//#include <vulkan/vulkan.h>
//#define GLFW_INCLUDE_GLU
//#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(NANOGUI_GLAD)
#if defined(NANOGUI_SHARED) && !defined(GLAD_GLAPI_EXPORT)
#define GLAD_GLAPI_EXPORT
#endif
#else
#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#else
#define GL_GLEXT_PROTOTYPES
#endif
#endif

#include <ponos/ponos.h>

#include <circe/colors/color.h>
#include <signal.h>

namespace circe::gl {

#ifdef GL_DEBUG

#define CHECK_GL(A)                                                            \
  {                                                                            \
    A;                                                                         \
    if (checkGL(__FILE__, __LINE__, __FUNCTION__, #A))                         \
      raise(SIGSEGV);                                                          \
  }
#define CHECK_GL_ERRORS                                                        \
  if (checkGL(__FILE__, __LINE__, __FUNCTION__))                               \
  raise(SIGSEGV);
#define CHECK_FRAMEBUFFER checkFramebuffer()
#else
#define CHECK_GL
#define CHECK_GL_ERRORS
#define CHECK_FRAMEBUFFER
#endif

inline void glfwError(int id, const char *description) {
  UNUSED_VARIABLE(id);
  std::cerr << description << std::endl;
}

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
/// \param file caller file
/// \param line_number caller line number
/// \param function caller function
/// \param line caller line
/// \return true if any OpenGL error occurred
bool checkGL(const char *file, int line_number, const char *function = nullptr,
             const char *line = nullptr);
///
/// \param error
/// \return string containing error description
std::string glErrorToString(GLenum error, bool description = true);

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

void glColor(Color c);

/// multiplies **t** to current OpenGL matrix
/// \param transform
void glApplyTransform(const ponos::Transform &transform);

ponos::Transform glGetProjectionTransform();

ponos::Transform glGetModelviewTransform();

ponos::Transform glGetMVPTransform();

} // namespace circe

#endif // CIRCE_UTILS_OPEN_GL_H
