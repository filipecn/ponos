#include <aergia/graphics/compute_shader.h>

namespace aergia {

ComputeShader::ComputeShader(const TextureAttributes &a, const TextureParameters &p, const char *source) {
  texture.reset(new Texture(a, p));
  programId = static_cast<GLuint>(std::max(0, ShaderManager::instance().loadFromText(source, GL_COMPUTE_SHADER)));
  { // query up the workgroups
    int work_grp_size[3], work_grp_inv;
    // maximum global work group (total work in a dispatch)
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_size[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_size[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_size[2]);
    printf("max global (total) work group size x:%i y:%i z:%i\n", work_grp_size[0],
           work_grp_size[1], work_grp_size[2]);
    // maximum local work group (one shader's slice)
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);
    printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n",
           work_grp_size[0], work_grp_size[1], work_grp_size[2]);
    // maximum compute shader invocations (x * y * z)
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
    printf("max computer shader invocations %i\n", work_grp_inv);
  }
}

ComputeShader::~ComputeShader() {}

bool ComputeShader::compute() {
  if (!ShaderManager::instance().useShader(programId))
    return false;
  texture->bindImage(GL_TEXTURE0);
  auto size = texture->size();
  glDispatchCompute(size[0], size[1], 1 );
  glMemoryBarrier( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );
  CHECK_GL_ERRORS;
  return true;
}

void ComputeShader::bindTexture(GLenum t) const {
  texture->bind(t);
}
} // aergia namespace
