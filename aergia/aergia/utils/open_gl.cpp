#include <aergia/utils/open_gl.h>

#include <cstdio>
#include <string>

namespace aergia {

void printShaderInfoLog(GLuint shader) {
  GLsizei infologLength = 0;
  int charsWritten = 0;
  GLchar *infoLog;

  CHECK_GL_ERRORS;

  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

  CHECK_GL_ERRORS;

  if (infologLength > 0) {
    infoLog = (GLchar *)malloc((size_t)infologLength);
    if (infoLog == NULL) {
      printf("ERROR: Could not allocate InfoLog buffer\n");
      exit(1);
    }
    glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
    printf("Shader InfoLog:\n%s\n\n", infoLog);
    std::cerr << "Shader InfoLog:\n" << infoLog << std::endl;
    free(infoLog);
    if (charsWritten) {
      exit(1);
    }
  }
  CHECK_GL_ERRORS;
}

void printProgramInfoLog(GLuint program) {
  int infologLength = 0;
  int charsWritten = 0;
  GLchar *infoLog;

  CHECK_GL_ERRORS;

  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

  CHECK_GL_ERRORS;

  if (infologLength > 0) {
    infoLog = (GLchar *)malloc(infologLength);
    if (infoLog == NULL) {
      printf("ERROR: Could not allocate InfoLog buffer\n");
      exit(1);
    }
    glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
    printf("Program InfoLog:\n%s\n\n", infoLog);
    free(infoLog);
  }
  CHECK_GL_ERRORS;
}

bool printOglError(const char *file, int line) {
  GLenum glErr;
  bool retCode = false;

  glErr = glGetError();
  while (glErr != GL_NO_ERROR) {
    std::cerr << "glError in file " << file << " @ line " << line << ": " << gluErrorString(glErr) << std::endl;
    retCode = true;
    glErr = glGetError();
  }
  return retCode;
}

bool checkFramebuffer() {
  std::string error;
  std::string name;
  switch (glCheckFramebufferStatus(GL_FRAMEBUFFER)) {
  case GL_FRAMEBUFFER_COMPLETE:
    return true;
  case GL_FRAMEBUFFER_UNDEFINED:
    error += "target is the default framebuffer, but the default framebuffer "
             "does not exist.";
    name += "GL_FRAMEBUFFER_UNDEFINED";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    error +=
        "any of the framebuffer attachment points are framebuffer incomplete.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    error += "the framebuffer does not have at least one image attached to it.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    error += "the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE "
             "for any color attachment point(s) named by GL_DRAW_BUFFERi.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    error += "GL_READ_BUFFER is not GL_NONE and the value of "
             "GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color "
             "attachment point named by GL_READ_BUFFER.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED:
    error += "the combination of internal formats of the attached images "
             "violates an implementation-dependent set of restrictions.";
    name += "GL_FRAMEBUFFER_UNSUPPORTED";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    error += "the value of GL_RENDERBUFFER_SAMPLES is not the same for all "
             "attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is "
             "the not same for all attached textures; or, if the attached "
             "images are a mix of renderbuffers and textures, the value of "
             "GL_RENDERBUFFER_SAMPLES does not match the value of "
             "GL_TEXTURE_SAMPLES.";
    error += " OR the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the "
             "same for all attached textures; or, if the attached images are a "
             "mix of renderbuffers and textures, the value of "
             "GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all "
             "attached textures.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
    error += "any framebuffer attachment is layered, and any populated "
             "attachment is not layered, or if all populated color attachments "
             "are not from textures of the same target.";
    name += "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
    break;
  default:
    break;
  }

  error += "\n";
  std::cout << "[CHECKFRAMBUFFER - " << name << "]\n" << error << std::endl;
  exit(1);

  return 0;
}

bool initGLEW() {
  glewExperimental = GL_TRUE;
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    return false;
  }
  fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
  return true;
}

void getGlVersion(int *major, int *minor) {
  const char *str = (const char *)glGetString(GL_VERSION);
  std::cout << ">>>>>>>>>>> " << str << std::endl;
  if ((str == NULL) || (sscanf(str, "%d.%d", major, minor) != 2)) {
    *major = *minor = 0;
    fprintf(stderr, "Invalid GL_VERSION format!!!\n");
  }
}

void glVertex(ponos::Point3 v) { glVertex3f(v.x, v.y, v.z); }

void glVertex(ponos::Point2 v) { glVertex2f(v.x, v.y); }

void glVertex(ponos::vec2 v) { glVertex2f(v.x, v.y); }

void glVertex(ponos::Point<float, 2> v) { glVertex2f(v[0], v[1]); }

void glColor(Color c) { glColor4f(c.r, c.g, c.b, c.a); }

void glApplyTransform(const ponos::Transform &transform) {
  float m[16];
  transform.matrix().column_major(m);
  glMultMatrixf(m);
}

ponos::Transform glGetProjectionTransform() {
  float m[16];
  glGetFloatv(GL_PROJECTION_MATRIX, m);
  return ponos::Transform(ponos::Matrix4x4(m, false));
}

ponos::Transform glGetModelviewTransform() {
  float m[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, m);
  return ponos::Transform(ponos::Matrix4x4(m, false));
}

ponos::Transform glGetMVPTransform() {
  // return glGetProjectionTransform() * glGetModelviewTransform();
  return glGetModelviewTransform() * glGetProjectionTransform();
}

} // aergia namespace
