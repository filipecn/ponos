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
#include <ponos/geometry/math_element.h>

namespace ponos {

/// Array of Structures
/// This class stores an array of structures that can be defined in runtime
class AoS {
public:
  struct FieldDescription {
    std::string name;
    u64 size{0};
    u64 offset{0};
    u32 component_count{1};
    DataType type{DataType::CUSTOM};
  };
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
  AoS &operator=(AoS &&other) noexcept;
  AoS &operator=(const AoS &other);
  template<typename T>
  AoS &operator=(std::vector<T> &&vector_data) {
    /// TODO: move operation is making a copy instead!
    if (!struct_size_) {
      spdlog::warn("[AoS] Fields must be previously registered.");
      return *this;
    }
    if (vector_data.size() * sizeof(T) % struct_size_ != 0)
      spdlog::warn("[AoS] Vector data with incompatible size.");
    delete[]data_;
    data_ = nullptr;
    size_ = vector_data.size() * sizeof(T) / struct_size_;
    data_ = new u8[size_ * struct_size_];
    std::memcpy(data_, vector_data.data(), size_ * struct_size_);
    return *this;
  }
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
        struct_size_,
        1,
        DataTypes::typeFrom<T>()
    };
#define MATCH_PONOS_TYPES(Type, DT, C) \
    if (std::is_base_of_v<ponos::MathElement<Type, C>, T>) { \
      d.component_count = C; \
      d.type = DataType::DT;\
    }
    MATCH_PONOS_TYPES(f32, F32, 2u)
    MATCH_PONOS_TYPES(f32, F32, 3u)
    MATCH_PONOS_TYPES(f32, F32, 4u)
    MATCH_PONOS_TYPES(f32, F32, 9u)
    MATCH_PONOS_TYPES(f32, F32, 16u)
#undef MATCH_PONOS_TYPES
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
  /// \return
  const std::vector<FieldDescription> &fields() const { return fields_; }
  /// \tparam T
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_ + i * struct_size_ + fields_[field_id].offset);
  }

  friend std::ostream &operator<<(std::ostream &o, const AoS &aos) {
#define PRINT_FIELD_VALUE(T, Type) \
        if (f.type == DataType::Type) { \
    const T *ptr = reinterpret_cast<T *>(aos.data_ + offset + f.offset); \
    for (u32 j = 0; j < f.component_count; ++j) \
      o << ptr[j] << ((j < f.component_count - 1) ?  " " : ""); \
    o << ") "; \
  }

    o << "AoS (struct count: " << aos.size_ << ") (struct size in bytes: "
      << aos.struct_size_ << ")\n";
    o << "fields: ";
    for (const auto &f : aos.fields_) {
      auto it = aos.field_id_map_.find(f.name);
      o << "field #" << it->second << " (" << f.name << " ): ";
      o << "\tbase data type: " << DataTypes::typeName(f.type) << "\n";
      o << "\tbase data size in bytes: " << f.size << "\n";
      o << "\tcomponent count: " << f.component_count << "\n";
      o << "field values:\n";
      u64 offset = 0;
      for (u64 i = 0; i < aos.size_; ++i) {
        o << "[" << i << "](";
        PRINT_FIELD_VALUE(i8, I8)
        PRINT_FIELD_VALUE(i16, I16)
        PRINT_FIELD_VALUE(i32, I32)
        PRINT_FIELD_VALUE(i64, I64)
        PRINT_FIELD_VALUE(u8, U8)
        PRINT_FIELD_VALUE(u16, U16)
        PRINT_FIELD_VALUE(u32, U32)
        PRINT_FIELD_VALUE(u64, U64)
        PRINT_FIELD_VALUE(f32, F32)
        PRINT_FIELD_VALUE(f64, F64)
        offset += aos.struct_size_;
      }
      o << std::endl;
    }
    return o;
#undef PRINT_FIELD_VALUE
  }

private:
  u64 size_{0}; //!< struct count
  u64 struct_size_{0}; //!< in bytes
  u8 *data_{nullptr};
  std::vector<FieldDescription> fields_;
  std::unordered_map<std::string, u64> field_id_map_;
};

}

#endif //PONOS_PONOS_PONOS_STORAGE_ARRAY_OF_STRUCTURES_H
