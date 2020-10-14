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
///\file array_of_structures.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-12-10
///
///\brief

#ifndef PONOS_PONOS_PONOS_STORAGE_ARRAY_OF_STRUCTURES_H
#define PONOS_PONOS_PONOS_STORAGE_ARRAY_OF_STRUCTURES_H

#include <ponos/common/defs.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace ponos {

/// Array of Structures
/// This class stores an array of structures that can be defined in runtime
class AoS {
public:
  template<typename T>
  class FieldAccessor {
    friend class AoS;
  public:
    T &operator[](u64 i) {
      return *reinterpret_cast<T *>(data_ + i * stride_ + offset_);
    }
  private:
    FieldAccessor(u8 *data, u64 stride, u64 offset) : data_{data}, stride_{stride}, offset_{offset} {}
    u8 *data_{nullptr};
    u64 stride_{0};
    u64 offset_{0};
  };
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  AoS();
  virtual ~AoS();
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  // ***********************************************************************
  //                             METHODS
  // ***********************************************************************
  /// \return
  [[nodiscard]] const u8 *data() const { return data_; }
  /// \param new_size in number of elements
  void resize(u64 new_size);
  template<typename T>
  FieldAccessor<T> field(const std::string &name) {
    auto it = field_id_map_.find(name);
    if (it == field_id_map_.end()) {
      spdlog::error("Field {} not found.", name);
      return FieldAccessor<T>(nullptr, 0, 0);
    }
    return FieldAccessor<T>(data_, struct_size_, fields_[it->second].offset);
  }
  /// \tparam T
  /// \param name
  /// \return
  template<typename T>
  u64 pushField(const std::string &name) {
    field_id_map_[name] = fields_.size();
    FieldDescription d = {
        name,
        sizeof(T),
        struct_size_
    };
    fields_.emplace_back(d);
    struct_size_ += d.size;
    return fields_.size() - 1;
  }
  inline const std::string &fieldName(u64 field_id) const { return fields_[field_id].name; }
  /// \return
  [[nodiscard]] inline u64 size() const { return size_; }
  /// \return
  [[nodiscard]] inline u64 memorySizeInBytes() const { return size_ * struct_size_; }
  /// \return
  [[nodiscard]] inline u64 stride() const { return struct_size_; }
  u64 offsetOf(const std::string &field_name) const;
  u64 offsetOf(u64 field_id) const;
  u64 sizeOf(const std::string &field_name) const;
  u64 sizeOf(u64 field_id) const;

  template<typename T>
  T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_ + i * struct_size_ + fields_[field_id].offset);
  }

private:
  u64 size_{0};
  u64 struct_size_{0};
  u8 *data_{nullptr};
  struct FieldDescription {
    std::string name;
    u64 size{0};
    u64 offset{0};
  };
  std::vector<FieldDescription> fields_;
  std::unordered_map<std::string, u64> field_id_map_;
};

}

#endif //PONOS_PONOS_PONOS_STORAGE_ARRAY_OF_STRUCTURES_H
