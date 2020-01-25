template <typename T>
Texture<T>::Texture(int w, int h, bool fromDevice, const T *data) : w(w), h(h) {
  resize(w, h);
  init();
  if (data) {
    if (fromDevice)
      CHECK_CUDA(cudaMemcpy(d_data, data, w * h * sizeof(T),
                            cudaMemcpyDeviceToDevice));
    else
      CHECK_CUDA(
          cudaMemcpy(d_data, data, w * h * sizeof(T), cudaMemcpyHostToDevice));
  }
}
template <typename T> Texture<T>::~Texture() { clear(); }

template <typename T> void Texture<T>::resize(int newWidth, int newHeight) {
  clear();
  w = newWidth;
  h = newHeight;
  init();
}

template <typename T> unsigned int Texture<T>::width() const { return w; }

template <typename T> unsigned int Texture<T>::height() const { return h; }

template <typename T> const cudaArray *Texture<T>::textureArray() const {
  return texArray;
}

template <typename T> T *Texture<T>::deviceData() { return d_data; }

template <typename T> const T *Texture<T>::deviceData() const { return d_data; }

template <typename T> void Texture<T>::updateTextureMemory() {
  if (d_data && texArray)
    CHECK_CUDA(cudaMemcpyToArray(texArray, 0, 0, d_data, w * h * sizeof(T),
                                 cudaMemcpyDeviceToDevice));
}

template <typename T> void Texture<T>::init() {
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
  CHECK_CUDA(cudaMalloc((void **)&d_data, sizeof(T) * w * h));
  CHECK_CUDA(cudaMallocArray(&texArray, &channelDesc, w, h));
}

template <typename T> void Texture<T>::clear() {
  if (d_data)
    cudaFree(d_data);
  if (texArray)
    cudaFreeArray(texArray);
}

template <typename T> void Texture<T>::copy(const Texture<T> &other) {
  if (w == other.w && h == other.h)
    CHECK_CUDA(cudaMemcpy(d_data, other.d_data, w * h * sizeof(T),
                          cudaMemcpyDeviceToDevice));
}

template <typename T> void fill(Texture<T> &texture, T value) {
  fillTexture(texture.deviceData(), value, texture.width(), texture.height());
  texture.updateTextureMemory();
}