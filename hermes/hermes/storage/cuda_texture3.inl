template <typename T>
Texture3<T>::Texture3(int w, int h, int d, bool fromDevice, T *data) {
  resize(w, h, d);
  init();
  if (data) {
    copyLinearToPitched(
        data, d_data,
        (fromDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, d);
  }
}
template <typename T> Texture3<T>::~Texture3() { clear(); }

template <typename T>
void Texture3<T>::resize(int newWidth, int newHeight, int newDepth) {
  clear();
  size.width = newWidth;
  size.height = newHeight;
  size.depth = newDepth;
  init();
}

template <typename T> unsigned int Texture3<T>::width() const {
  return size.width;
}

template <typename T> unsigned int Texture3<T>::height() const {
  return size.height;
}

template <typename T> unsigned int Texture3<T>::depth() const {
  return size.depth;
}

template <typename T> const cudaArray *Texture3<T>::textureArray() const {
  return texArray;
}

template <typename T> T *Texture3<T>::deviceData() {
  return reinterpret_cast<T *>(d_data.ptr);
}

template <typename T> const T *Texture3<T>::deviceData() const {
  return reinterpret_cast<const T *>(d_data.ptr);
}

template <typename T> cudaPitchedPtr Texture3<T>::pitchedData() {
  return d_data;
}

template <typename T> size_t Texture3<T>::pitch() const { return d_data.pitch; }

template <typename T> void Texture3<T>::updateTextureMemory() {
  if (d_data.ptr && texArray) {
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = d_data;
    copyParams.dstArray = texArray;
    copyParams.extent = size;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
  }
}

template <typename T> void Texture3<T>::init() {
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  cudaExtent extent =
      make_cudaExtent(size.width * sizeof(T), size.height, size.depth);
  CUDA_CHECK(cudaMalloc3D(&d_data, extent));
  CUDA_CHECK(cudaMalloc3DArray(&texArray, &channelDesc, size));
}

template <typename T> void Texture3<T>::clear() {
  if (d_data.ptr)
    cudaFree(d_data.ptr);
  if (texArray)
    cudaFreeArray(texArray);
}

template <typename T> void Texture3<T>::copy(const Texture3<T> &other) {
  if (size.width == other.size.width && size.height == other.size.height &&
      size.depth == other.size.depth) {
    cudaMemcpy3DParms p = {0};
    p.srcPtr.ptr = other.d_data.ptr;
    p.srcPtr.pitch = size.height * sizeof(T);
    p.srcPtr.xsize = size.width;
    p.srcPtr.ysize = size.height;
    p.dstPtr.ptr = d_data.ptr;
    p.dstPtr.pitch = d_data.pitch;
    p.dstPtr.xsize = size.width;
    p.dstPtr.ysize = size.height;
    p.extent.width = size.height * sizeof(T);
    p.extent.height = size.height;
    p.extent.depth = size.depth;
    p.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

template <typename T> void fill(Texture3<T> &texture, T value) {
  fillTexture(texture.pitchedData(), value, texture.width(), texture.height(),
              texture.depth());
  texture.updateTextureMemory();
}