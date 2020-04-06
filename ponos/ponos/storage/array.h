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

/// Holds a linear memory area representing a 2-dimensional array of
/// ``size.width`` * ``size.height`` elements.
///
/// - Considering ``size.height`` rows of ``size.width`` elements, data is
/// laid out in memory in **row major** fashion.
///
/// - It is also possible to set _row level_ memory alignment via a custom size
/// of allocated memory per row, called pitch size. The minimal size of pitch is
/// ``size.width``*``sizeof(T)``.
///
/// - The methods use the convention of ``i`` and ``j`` indices, representing
/// _column_ and _row_ indices respectively. ``i`` accesses the first
/// dimension (``size.width``) and ``j`` accesses the second dimension
/// (``size.height``).
/// \verbatim embed:rst:leading-slashes"
///   .. note::
///     This index convention is the **opposite** of some mathematical forms
///     where matrix elements are indexed by the i-th row and j-th column.
/// \endverbatim
/// - Array2 provides a convenient way to access its elements:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       for(auto e : my_array2) {
///         e.value = 0; // element value access
///         e.index; // element index
///       }
/// \endverbatim
/// \tparam T data type
template <class T> class Array2 {
public:
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  Array2() = default;
  /// pitch is set to ``size.width`` * ``sizeof(T)``
  /// \param size dimensions (in elements count)
  Array2(const size2 &size) : pitch_(size.width * sizeof(T)), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// \param size dimensions (in elements count)
  /// \param pitch memory size occupied by a single row (in bytes)
  explicit Array2(const size2 &size, size_t pitch)
      : pitch_(pitch), size_(size) {
    data_ = new char[pitch_ * size.height];
  }
  /// Copy constructor
  /// \param other **[in]** const reference to other Array2 object
  Array2(const Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  Array2(const Array2 &&other) = delete;
  /// Copy constructor
  /// \param other **[in]** reference to other Array2 object
  Array2(Array2 &other) : Array2(other.size_, other.pitch_) {
    memcpy(data_, other.data_, memorySize());
  }
  /// Assign constructor
  /// \param other **[in]** temporary Array2 object
  Array2(Array2 &&other) noexcept
      : pitch_(other.pitch_), size_(other.size_), data_(other.data_) {
    other.data_ = nullptr;
  }
  /// Constructs an Array2 from a std vector matrix
  /// \param linear_vector **[in]** data matrix
  Array2(const std::vector<std::vector<T>> &linear_vector) {
    resize(size2(linear_vector.size(), linear_vector[0].size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = linear_vector[ij.i][ij.j];
  }
  /// Initialization list constructor
  ///
  /// - Inner lists represent rows.
  /// \param list **[in]** data list
  /// \verbatim embed:rst:leading-slashes
  ///    **Example**::
  ///
  ///       ponos::Array2<i32> a = {{1,2},{3,4}};
  /// \endverbatim
  Array2(std::initializer_list<std::initializer_list<T>> list) {
    resize(size2(list.size(), list.begin()[0].size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = list.begin()[ij.i].begin()[ij.j];
  }
  ///
  virtual ~Array2() {
    if (data_)
      delete[](char *) data_;
  }
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  /// \param linear_vector **[in]**
  /// \return Array2<T>&
  Array2<T> &operator=(const std::vector<std::vector<T>> &linear_vector) {
    resize(size2(linear_vector.size(), linear_vector[0].size()));
    for (auto ij : Index2Range<i32>(size_))
      (*this)[ij] = linear_vector[ij.i][ij.j];
  }
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
  /// Assign ``value`` to all elements
  /// \param value assign value
  /// \return *this
  Array2 &operator=(T value) {
    if (!data_)
      data_ = new char[pitch_ * size_.height];
    for (index2 ij : Index2Range<i32>(size_))
      (*this)[ij] = value;
    return *this;
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij ``ij.i`` for column and ``ij.j`` for row
  /// \return reference to element at ``ij`` position
  T &operator[](index2 ij) {
    return (T &)(*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``ij`` is out of bounds.
  /// \endverbatim
  /// \param ij ``ij.i`` for column and ``ij.j`` for row
  /// \return const reference to element at ``ij`` position
  const T &operator[](index2 ij) const {
    return (T &)(*((char *)data_ + ij.j * pitch_ + ij.i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` or ``j`` are out of bounds.
  /// \endverbatim
  /// \param i **[in]** column index
  /// \param j **[in]** row index
  /// \return T& reference to element in row ``i`` and column ``j``
  T &operator()(u32 i, u32 j) {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  /// \verbatim embed:rst:leading-slashes
  ///    .. warning::
  ///       This method does **not** check if ``i`` or ``j`` are out of bounds.
  /// \endverbatim
  /// \param i **[in]** column index
  /// \param j **[in]** row index
  /// \return const reference to element at ``ij`` position
  const T &operator()(u32 i, u32 j) const {
    return (T &)(*((char *)data_ + j * pitch_ + i * sizeof(T)));
  }
  // ***********************************************************************
  //                         GETTERS & SETTERS
  // ***********************************************************************
  /// Changes the dimensions
  /// \verbatim embed:rst:leading-slashes
  ///    .. note::
  ///       All previous data is erased.
  /// \endverbatim
  /// \param new_size new row and column counts
  void resize(const size2 &new_size) {
    if (data_)
      delete[](char *) data_;
    pitch_ = std::max(pitch_, sizeof(T) * new_size.width);
    size_ = new_size;
    data_ = new char[pitch_ * new_size.height];
  }
  /// Computes actual memory usage
  /// \return memory usage (in bytes)
  size_t memorySize() const { return size_.height * pitch_; }
  /// \return dimensions (in elements count)
  size2 size() const { return size_; }
  /// \return pitch size (in bytes)
  size_t pitch() const { return pitch_; }
  /// \return const pointer to raw data (**row major**)
  const T *data() const { return (const T *)data_; }
  /// \return pointer to raw data (**row major**)
  T *data() { return (T *)data_; }
  // ***********************************************************************
  //                            METHODS
  // ***********************************************************************
  /// Copies data from another Array2
  ///
  /// - This gets resized if necessary.
  /// \param other **[in]**
  void copy(const Array2 &other) {
    pitch_ = other.pitch_;
    size_ = other.size_;
    resize(size_);
    memcpy(data_, other.data_, memorySize());
  }
  /// Checks if ``ij`` is not out of bounds
  /// \param ij position index
  /// \return ``true`` if position can be accessed
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

template <typename T>
std::ostream &operator<<(std::ostream &os, const Array2<T> &array) {
  os << "Array2[" << array.size() << "]\n\t\t";
  for (u32 i = 0; i < array.size().height; ++i)
    os << "[," << i << "]\t";
  os << std::endl;
  for (u32 i = 0; i < array.size().width; ++i) {
    os << "\t[" << i << ",]\t";
    for (u32 j = 0; j < array.size().height; ++j)
      if (std::is_same<T, u8>())
        os << (int)array[index2(i, j)] << "\t";
      else
        os << array[index2(i, j)] << "\t";
    os << std::endl;
  }
  return os;
}

using array2d = Array2<f64>;
using array2f = Array2<f32>;
using array2i = Array2<i32>;
using array2u = Array2<u32>;

} // namespace ponos

#endif // PONOS_STORAGE_ARRAY_H