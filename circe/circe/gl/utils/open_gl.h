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
#include <csignal>

namespace circe::gl {

class OpenGL final {
public:
  [[nodiscard]] static u64 dataSizeInBytes(GLuint data_type) {
    switch (data_type) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE: return 1;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
    case GL_HALF_FLOAT: return 2;
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_FIXED:
    case GL_FLOAT:return 4;
    case GL_DOUBLE: return 8;
    default: return 0;
    }
    return 0;
  }
  template<typename T>
  static GLenum dataTypeEnum() {
    if (std::is_same_v<T, i32>)
      return GL_INT;
    if (std::is_same_v<T, u32>)
      return GL_UNSIGNED_INT;
    if (std::is_same_v<T, f32>)
      return GL_FLOAT;
    if (std::is_same_v<T, f64>)
      return GL_DOUBLE;
    if (std::is_same_v<T, u8>)
      return GL_UNSIGNED_BYTE;
    if (std::is_same_v<T, i8>)
      return GL_BYTE;
    if (std::is_same_v<T, i16>)
      return GL_SHORT;
    if (std::is_same_v<T, u16>)
      return GL_UNSIGNED_SHORT;
//    if (std::is_same_v<T, f16>)
//      return GL_HALF_FLOAT;
    return GL_FIXED;
  }

};

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
