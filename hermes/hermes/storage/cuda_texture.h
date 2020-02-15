/*
 * Copyright (c) 2019 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_STORAGE_TEXTURE_H
#define HERMES_STORAGE_TEXTURE_H

#include <hermes/common/cuda.h>
#include <hermes/storage/cuda_storage_utils.h>
#include <hermes/storage/cuda_texture_kernels.h>
#include <iomanip>
#include <ponos/numeric/grid.h>
#include <type_traits>
#include <vector>

namespace hermes {

enum class TextureReadMode { ELEMENT, NORMALIZED };

struct TextureDescriptor {
  TextureDescriptor() {}
  TextureDescriptor(ponos::AddressMode amode) {
    for (int i = 0; i < 3; i++)
      addressMode[i] = amode;
  }
  ponos::AddressMode addressMode[3] = {ponos::AddressMode::BORDER,
                                       ponos::AddressMode::BORDER,
                                       ponos::AddressMode::BORDER};
  ponos::FilterMode filterMode = ponos::FilterMode::LINEAR;
  TextureReadMode readMode = TextureReadMode::ELEMENT;
  bool normalizedCoordinates = false;
  bool sRGB = false;
};

namespace cuda {

inline cudaTextureAddressMode cudaValue(ponos::AddressMode addressMode) {
  switch (addressMode) {
  case ponos::AddressMode::BORDER:
    return cudaAddressModeBorder;
  case ponos::AddressMode::WRAP:
    return cudaAddressModeWrap;
  case ponos::AddressMode::CLAMP_TO_EDGE:
    return cudaAddressModeClamp;
  }
  return cudaAddressModeMirror;
}

inline cudaTextureFilterMode cudaValue(ponos::FilterMode filterMode) {
  switch (filterMode) {
  case ponos::FilterMode::LINEAR:
    return cudaFilterModeLinear;
  case ponos::FilterMode::POINT:
    return cudaFilterModePoint;
  }
  return cudaFilterModeLinear;
}

inline cudaTextureReadMode cudaValue(TextureReadMode readMode) {
  switch (readMode) {
  case TextureReadMode::ELEMENT:
    return cudaReadModeElementType;
  case TextureReadMode::NORMALIZED:
    return cudaReadModeNormalizedFloat;
  }
  return cudaReadModeElementType;
}

///
/// \tparam T
template <typename T> class Texture {
public:
  Texture() = default;
  virtual ~Texture();
  /// \param w texture width (in texels)
  /// \param h texture height (in texels)
  /// \param texDesc texture parameters
  /// \param fromDevice **[optional]** initial data location
  /// \param data **[optional]** initial data
  Texture(int w, int h, bool fromDevice = false, const T *data = nullptr);
  /// Clear texture memory and allocates a new one with new size
  /// \param newWidth new texture width (in texels)
  /// \param newHeight new texture height (in texels)
  void resize(int newWidth, int newHeight);
  /// \return unsigned int texture width (in texels)
  unsigned int width() const;
  /// \return unsigned int texture height (in texels)
  unsigned int height() const;
  /// \return const cudaArray* texture memory pointer
  const cudaArray *textureArray() const;
  /// \return T* device's global memory data
  T *deviceData();
  /// \return const T* device's global memory data
  const T *deviceData() const;
  /// Copies data from global memory (read/write) to texture memory (read only)
  void updateTextureMemory();
  void copy(const Texture<T> &other);

private:
  void init();
  void clear();

  unsigned int w = 0, h = 0;
  T *d_data = nullptr;
  cudaArray *texArray = nullptr;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Texture<T> &tex) {
  std::vector<T> data(tex.width() * tex.height());
  cudaMemcpy(&data[0], tex.deviceData(), tex.width() * tex.height() * sizeof(T),
             cudaMemcpyDeviceToHost);
  for (int y = tex.height() - 1; y >= 0; y--) {
    os << "l " << y << ": ";
    for (int x = 0; x < tex.width(); x++)
      if (std::is_same<T, char>::value || std::is_same<T, unsigned char>::value)
        os << (int)data[y * tex.width() + x] << "\t";
      else
        os << std::setprecision(6) << data[y * tex.width() + x] << "\t";
    os << std::endl;
  }
  return os;
}

/// Set all texture textel values
/// \tparam T data type
/// \param texture
/// \param value
template <typename T> void fill(Texture<T> &texture, T value);

/// 3D Texture data
/// \tparam T
template <typename T> class Texture3 {
public:
  Texture3() = default;
  virtual ~Texture3() { clear(); }
  /// \param w texture width (in texels)
  /// \param h texture height (in texels)
  /// \param d texture depth (in texels)
  /// \param texDesc texture parameters
  /// \param fromDevice **[optional]** initial data location
  /// \param data **[optional]** initial data
  Texture3(int w, int h, int d, bool fromDevice = false, T *data = nullptr) {
    resize(w, h, d);
    init();
    if (data) {
      copyLinearToPitched(
          d_data, data,
          (fromDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, d);
    }
  }
  /// Clear texture memory and allocates a new one with new size
  /// \param newWidth new texture width (in texels)
  /// \param newHeight new texture height (in texels)
  /// \param newDepth new texture depth (in texels)
  void resize(int newWidth, int newHeight, int newDepth) {
    clear();
    size.width = newWidth;
    size.height = newHeight;
    size.depth = newDepth;
    init();
  }
  /// \return unsigned int texture width (in texels)
  unsigned int width() const { return size.width; }
  /// \return unsigned int texture height (in texels)
  unsigned int height() const { return size.height; }
  /// \return unsigned int texture depth (in texels)
  unsigned int depth() const { return size.depth; }
  /// \return const cudaArray* texture memory pointer
  const cudaArray *textureArray() const { return texArray; }
  /// \return T* device's global memory data
  T *deviceData() { return reinterpret_cast<T *>(d_data.ptr); }
  cudaPitchedPtr pitchedData() { return d_data; }
  ///
  size_t pitch() const { return d_data.pitch; }
  /// \return const T* device's global memory data
  const T *deviceData() const {
    return reinterpret_cast<const T *>(d_data.ptr);
  }
  /// allocates texture memory on device
  void allocateTextureMemory() {
    releaseTextureMemory();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CHECK_CUDA(cudaMalloc3DArray(&texArray, &channelDesc, size));
  }
  /// free texture memory on device
  void releaseTextureMemory() {
    if (texArray)
      cudaFreeArray(texArray);
    texArray = nullptr;
  }
  /// Copies data from global memory (read/write) to texture memory (read only)
  void updateTextureMemory() {
    if (d_data.ptr && texArray) {
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr = d_data;
      copyParams.dstArray = texArray;
      copyParams.extent = size;
      copyParams.kind = cudaMemcpyDeviceToDevice;
      CHECK_CUDA(cudaMemcpy3D(&copyParams));
    }
  }
  void copy(const Texture3<T> &other) {
    if (size.width == other.size.width && size.height == other.size.height &&
        size.depth == other.size.depth)
      copyPitchedToPitched<T>(d_data, other.d_data, cudaMemcpyDeviceToDevice,
                              size.depth);
  }

private:
  void init() {
    cudaExtent extent =
        make_cudaExtent(size.width * sizeof(T), size.height, size.depth);
    CHECK_CUDA(cudaMalloc3D(&d_data, extent));
  }
  void clear() {
    if (d_data.ptr)
      cudaFree(d_data.ptr);
    if (texArray)
      cudaFreeArray(texArray);
  }

  cudaExtent size;
  cudaPitchedPtr d_data;
  cudaArray *texArray = nullptr;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, Texture3<T> &tex) {
  std::vector<T> data(tex.width() * tex.height() * tex.depth());
  copyPitchedToLinear(data.data(), tex.pitchedData(), cudaMemcpyDeviceToHost,
                      tex.depth());
  std::cerr << "3D Texture (" << tex.width() << " x " << tex.height() << " x "
            << tex.depth() << ")\n";
  for (int z = 0; z < tex.depth(); z++)
    for (int y = tex.height() - 1; y >= 0; y--) {
      os << "l[" << y << "] d[" << z << "]: ";
      for (int x = 0; x < tex.width(); x++)
        if (std::is_same<T, char>::value ||
            std::is_same<T, unsigned char>::value)
          os << (int)data[z * tex.width() * tex.height() + y * tex.width() + x]
             << "\t";
        else
          os << std::setprecision(6)
             << data[z * tex.width() * tex.height() + y * tex.width() + x]
             << "\t";
      os << std::endl;
    }
  return os;
}

/// Set all texture textel values
/// \tparam T data type
/// \param texture
/// \param value
template <typename T> void fill(Texture3<T> &texture, T value) {
  fillTexture(texture.pitchedData(), value, texture.width(), texture.height(),
              texture.depth());
  texture.updateTextureMemory();
}

#include "cuda_texture.inl"
// #include "cuda_texture3.inl"

} // namespace cuda

} // namespace hermes

#endif // HERMES_STORAGE_TEXTURE_H
