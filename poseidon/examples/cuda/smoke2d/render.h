#include <circe/circe.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <hermes/common/cuda.h>
#include <hermes/hermes.h>

void renderDensity(unsigned int w, unsigned int h,
                   const hermes::cuda::Texture<float> &in, unsigned int *out);

class CudaOpenGLInterop {
public:
  CudaOpenGLInterop(unsigned int w, unsigned int h) : width(w), height(h) {
    using namespace hermes::cuda;
    unsigned int size_tex_data = sizeof(GLubyte) * width * height * 4;
    CUDA_CHECK(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    circe::TextureAttributes ta;
    ta.width = width;
    ta.height = height;
    ta.internalFormat = GL_RGBA8;
    ta.format = GL_RGBA;
    ta.type = GL_UNSIGNED_BYTE;
    ta.target = GL_TEXTURE_2D;
    circe::TextureParameters tp;
    tp[GL_TEXTURE_MIN_FILTER] = GL_NEAREST;
    tp[GL_TEXTURE_MAG_FILTER] = GL_NEAREST;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }

  ~CudaOpenGLInterop() {
    if (cuda_dev_render_buffer)
      cudaFree(cuda_dev_render_buffer);
  }

  void sendToTexture() {
    using namespace hermes::cuda;
    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr,
                                                     cuda_tex_resource, 0, 0));

    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    CUDA_CHECK(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer,
                                 size_tex_data, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
  }

  void bindTexture(GLenum t) { texture.bind(t); }

  unsigned int *bufferPointer() {
    return (unsigned int *)cuda_dev_render_buffer;
  }

private:
  circe::Texture texture;
  unsigned int width, height;
  void *cuda_dev_render_buffer = nullptr;
  struct cudaGraphicsResource *cuda_tex_resource = nullptr;
};