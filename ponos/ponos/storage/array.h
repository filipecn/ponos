/// Copyright (c) 2020, FilipeCN.
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
///\file array.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-29
///
///\brief

#ifndef PONOS_STORAGE_ARRAY_H
#define PONOS_STORAGE_ARRAY_H

#include <ponos/common/index.h>
#include <ponos/common/size.h>

namespace ponos {

/// Auxiliary class to iterate an Array2 inside a for loop.
/// Ex: for(auto e : array) { e.value = x; }
///\tparam T Array2 data type
template <typename T> class Array2Iterator {
public:
  class Element {
  public:
    Element(T &v, const index2 &ij) : value(v), index(ij) {}
    T &value;
    const index2 index;
  };
  Array2Iterator(T *data, const size2 &size, size_t pitch, const index2 &ij)
      : size_(size), data_(data), pitch_(pitch), i(ij.i), j(ij.j) {}
  size2 size() const { return size_; }
  Array2Iterator &operator++() {
    i++;
    if (i >= static_cast<i64>(size_.width)) {
      i = 0;
      j++;
      if (j >= static_cast<i64>(size_.height)) {
        i = j = -1;
      }
    }
    return *this;
  }
  Element operator*() {
    return Element((T &)(*((char *)data_ + j * pitch_ + i * sizeof(T))),
                   index2(i, j));
  }
  bool operator==(const Array2Iterator &other) {
    return size_ == other.size_ && data_ == other.data_ && i == other.i &&
           j == other.j && pitch_ == other.pitch_;
  }
  bool operator!=(const Array2Iterator &other) {
    return size_ != other.size_ || data_ != other.data_ || i != other.i ||
           j != other.j || pitch_ != other.pitch_;
  }

private:
  size2 size_;
  T *data_ = nullptr;
  size_t pitch_ = 0;
  int i = 0, j = 0;
};

/// Holds a linear memory area representing a 2-dimensional array. It provides
/// memory alignment via a custom size of allocated memory per row (called pitch
/// size).
/// \tparam T data type
template <class T> class Array2 {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Array2() = default;
  /// \param size dimensions (in element count)
  Array2(const size2 &size) : pitch_(size.width * sizeof(T)), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// \param size dimensions (in element count)
  /// \param pitch in bytes
  explicit Array2(const size2 &size, size_t pitch)
      : pitch_(pitch), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// \param other **[in]**
  Array2(const Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  Array2(const Array2 &&other) = delete;
  /// \param other **[in]**
  Array2(Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  /// \param other **[in]**
  Array2(Array2 &&other) noexcept
      : pitch_(other.pitch_), size_(other.size_), data_(other.data_) {
    other.data_ = nullptr;
  }
  ///
  virtual ~Array2() {
    if (data_)
      delete[](char *) data_;
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// \param other **[in]**
  /// \return Array2<T>&
  Array2<T> &operator=(const Array2<T> &other) {
    size_ = other.size_;
    pitch_ = other.pitch_;
    resize(size_);
    memcpy(data_, other.data_, other.memorySize());
    return *this;
  }
  /// \param other **[in]**
  /// \return Array2<T>&
  Array2<T> &operator=(Array2<T> &&other) {
    size_ = other.size_;
    pitch_ = other.pitch_;
    data_ = other.data_;
    other.data_ = nullptr;
    return *this;
  }
  /// Assign value to all data
  /// \param value assign value
  /// \return *this
  Array2 &operator=(T value) {
    if (!data_)
      data_ = new char[pitch_ * size_.height];
    for (index2 ij : Index2Range<i32>(size_))
      (*this)[ij] = value;
    return *this;
  }
  /// \param ij (0 <= ij < size)
  /// \return reference to element at ij position
  T &operator[](index2 ij) {
    return (T &)(*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \param ij (0 <= ij < size)
  /// \return const reference to element at ij position
  const T &operator[](index2 ij) const {
    return (T &)(*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \param i **[in]**
  /// \param j **[in]**
  /// \return T&
  T &operator()(u32 i, u32 j) {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  /// \param ij (0 <= ij < size)
  /// \return const reference to element at ij position
  const T &operator()(u32 i, u32 j) const {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// \param new_size
  void resize(const size2 &new_size) {
    if (data_)
      delete[](char *) data_;
    pitch_ = std::max(pitch_, sizeof(T) * new_size.width);
    size_ = new_size;
    data_ = new char[pitch_ * new_size.height];
  }
  /// \return memory usage (in bytes)
  size_t memorySize() const { return size_.height * pitch_; }
  /// \return dimensions (in element count)
  size2 size() const { return size_; }
  /// \return pitch size (in bytes)
  size_t pitch() const { return pitch_; }
  /// \return const pointer to raw data
  const T *data() const { return (const T *)data_; }
  /// \return pointer to raw data
  T *data() { return (T *)data_; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  void copy(const Array2 &other) {
    pitch_ = other.pitch_;
    size_ = other.size_;
    resize(size_);
    memcpy(data_, other.data_, memorySize());
  }
  /// Checks if ij is a safe position to access
  /// \param ij position index
  /// \return true if position can be accessed
  bool stores(const index2 &ij) const {
    return ij.i >= 0 &&
           static_cast<i64>(ij.i) < static_cast<i64>(size_.width) &&
           ij.j >= 0 && static_cast<i64>(ij.j) < static_cast<i64>(size_.height);
  }
  Array2Iterator<T> begin() {
    return Array2Iterator<T>((T *)data_, size_, pitch_, index2(0, 0));
  }
  Array2Iterator<T> end() {
    return Array2Iterator<T>((T *)data_, size_, pitch_, index2(-1, -1));
  }

private:
  size_t pitch_{0};
  size2 size_{};
  void *data_ = nullptr;
};

} // namespace ponos

#endif // PONOS_STORAGE_ARRAY_H