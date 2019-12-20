/// Copyright (c) 2019, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file memory_block.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-12-12
///
///\brief

#ifndef PONOS_STORAGE_MEMORY_BLOCK_H
#define PONOS_STORAGE_MEMORY_BLOCK_H

#include <ponos/common/size.h>
#include <ponos/common/index.h>

namespace ponos {

/// Holds a linear memory area representing a 2-dimensional array. It provides
/// memory alignment via a custom size of allocated memory per row (called pitch
/// size).
/// \tparam T data type
template<class T>
class MemoryBlock2 {
public:
  MemoryBlock2() = default;
  /// \param size dimensions (in element count)
  explicit MemoryBlock2(const size2 &size) : pitch_(size.width * sizeof(T)),
                                             size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// \param size dimensions (in element count)
  /// \param pitch in bytes
  explicit MemoryBlock2(const size2 &size, size_t pitch)
      : pitch_(pitch), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  MemoryBlock2(const MemoryBlock2 &other) : MemoryBlock2(other.size_,
                                                         other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  MemoryBlock2(const MemoryBlock2 &&other) = delete;
  MemoryBlock2(MemoryBlock2 &other) : MemoryBlock2(other.size_,
                                                   other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  MemoryBlock2(MemoryBlock2 &&other) noexcept
      : pitch_(other.pitch_), size_(other.size_), data_(other.data_) {
    other.data_ = nullptr;
  }
  virtual ~MemoryBlock2() {
    if (data_)
      delete[] data_;
  }
  void copy(const MemoryBlock2 &other) {
    pitch_ = other.pitch_;
    size_ = other.size_;
    if (!data_)
      data_ = new char[pitch_ * size_.height];
    memcpy(data_, other.data_, memorySize());
  }
  /// \param new_size
  void resize(const size2 &new_size) {
    if (data_)
      delete[] data_;
    size_ = new_size;
    pitch_ = new_size.width * sizeof(T);
    data_ = new char[pitch_ * new_size.height];
  }
  /// \param ij (0 <= ij < size)
  /// \return reference to element at ij position
  T &operator()(index2 ij) {
    return (T &) (*((char *) data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \param ij (0 <= ij < size)
  /// \return const reference to element at ij position
  const T &operator()(index2 ij) const {
    return (T &) (*((char *) data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \return memory usage (in bytes)
  size_t memorySize() const { return size_.height * pitch_; }
  /// \return dimensions (in element count)
  size2 size() const { return size_; }
  /// \return pitch size (in bytes)
  size_t pitch() const { return pitch_; }
  /// \return const pointer to raw data
  const T *data() const { return data_; }
  /// \return pointer to raw data
  T *data() { return data_; }
  /// Checks if ij is a safe position to access
  /// \param ij position index
  /// \return true if position can be accessed
  bool stores(const index2 &ij) const {
    return ij.i >= 0 && ij.i < size_.width && ij.j >= 0 && ij.j < size_.height;
  }
  /// Assign value to all data
  /// \param value assign value
  /// \return *this
  MemoryBlock2& operator=(const T& value) {
    if(!data_)
      data_ = new char[pitch_ * size_.height];
    for(index2 ij : Index2Range<i32>(size_))
      (*this)(ij) = value;
    return *this;
  }
private:
  size_t pitch_ = 0;
  size2 size_{};
  void *data_ = nullptr;
};

} // ponos namespace

#endif //PONOS_STORAGE_MEMORY_BLOCK_H