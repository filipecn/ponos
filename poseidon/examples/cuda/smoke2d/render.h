#include <circe/circe.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <hermes/common/cuda.h>
#include <hermes/hermes.h>

void renderScalarGradient(unsigned int w, unsigned int h,
                          const hermes::cuda::Texture<float> &in,
                          unsigned int *out, float minValue, float maxValue,
                          hermes::cuda::Color a, hermes::cuda::Color b);

void renderDensity(unsigned int w, unsigned int h,
                   const hermes::cuda::Texture<float> &in, unsigned int *out);

void renderSolids(unsigned int w, unsigned int h,
                  const hermes::cuda::Texture<unsigned char> &in,
                  unsigned int *out);

template <typename T> class CudaGLTextureInterop {
public:
  CudaGLTextureInterop(const circe::Texture &texture, T *d_data = nullptr) {
    using namespace hermes::cuda;
    size = texture.size();
    target = texture.target();
    textureObjectId = texture.textureObjectId();
    if (d_data)
      d_buffer = d_data;
    else {
      freeBufferOnDestroy = true;
      CUDA_CHECK(
          cudaMalloc((void **)&d_buffer, size.x * size.y * size.z * sizeof(T)));
    }
    // Register this texture with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cudaResource, textureObjectId, texture.target(),
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }
  ~CudaGLTextureInterop() {
    if (d_buffer && freeBufferOnDestroy)
      cudaFree(d_buffer);
  }
  void sendToTexture() {
    using namespace hermes::cuda;
    // Map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaResource,
                                                     0, 0));
    CUDA_CHECK(cudaMemcpyToArray(texture_ptr, 0, 0, d_buffer,
                                 size.x * size.y * size.z * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
  }

  void bind(GLenum t) {
    glActiveTexture(t);
    glBindTexture(target, textureObjectId);
  }

  T *bufferPointer() { return d_buffer; }

private:
  GLenum target = GL_TEXTURE_3D;
  GLuint textureObjectId = 0;
  ponos::uivec3 size;
  T *d_buffer = nullptr;
  bool freeBufferOnDestroy = false;
  struct cudaGraphicsResource *cudaResource = nullptr;
};

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
    // tp[GL_TEXTURE_WRAP_S] = GL_CLAMP_TO_BORDER;
    // tp[GL_TEXTURE_WRAP_T] = GL_CLAMP_TO_BORDER;
    // tp[GL_TEXTURE_WRAP_R] = GL_CLAMP_TO_BORDER;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }

  CudaOpenGLInterop(unsigned int w, unsigned int h, unsigned int d)
      : width(w), height(h), depth(d) {
    using namespace hermes::cuda;
    unsigned int size_tex_data = width * height * depth * sizeof(float);
    CUDA_CHECK(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    circe::TextureAttributes ta;
    ta.width = width;
    ta.height = height;
    ta.depth = depth;
    ta.internalFormat = GL_RED;
    ta.format = GL_RED;
    ta.type = GL_FLOAT;
    ta.target = GL_TEXTURE_3D;
    circe::TextureParameters tp;
    tp.target = GL_TEXTURE_3D;
    tp[GL_TEXTURE_MIN_FILTER] = GL_LINEAR;
    tp[GL_TEXTURE_MAG_FILTER] = GL_LINEAR;
    tp[GL_TEXTURE_WRAP_S] = GL_CLAMP_TO_BORDER;
    tp[GL_TEXTURE_WRAP_T] = GL_CLAMP_TO_BORDER;
    tp[GL_TEXTURE_WRAP_R] = GL_CLAMP_TO_BORDER;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_3D,
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

    int num_texels = width * height * depth;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    CUDA_CHECK(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer,
                                 size_tex_data, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
  }

  void bindTexture(GLenum t) { texture.bind(t); }
  template <typename T> T *bufferPointer() {
    return (T *)cuda_dev_render_buffer;
  }

private:
  circe::Texture texture;
  unsigned int width = 0, height = 0, depth = 1;
  void *cuda_dev_render_buffer = nullptr;
  struct cudaGraphicsResource *cuda_tex_resource = nullptr;
};