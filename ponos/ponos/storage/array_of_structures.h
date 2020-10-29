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

/// forward declaration of AoS
class AoS;

/// Describes the structure that is stored in an array of structures
class StructDescriptor {
public:
  friend class AoS;
  friend class AoSAccessor;
  friend class AoSConstAccessor;
  /// Field description
  /// \verbatim embed:rst:leading-slashes
  ///    **Example**::
  ///       Suppose a single field named "field_a" and defined as float[3].
  ///       Then name is "field_a", size is 3 * sizeof(float), offset is 0,
  ///       component_count is 3 and type is DataType::F32
  /// \endverbatim
  struct Field {
    std::string name; //!< field name
    u64 size{0}; //!< field size in bytes
    u64 offset{0}; //!< field offset in bytes inside structure
    u32 component_count{1}; //!< component count of data type
    DataType type{DataType::CUSTOM};
  };
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  StructDescriptor() = default;
  ~StructDescriptor() = default;
  // ***********************************************************************
  //                            OPERATORS
  // ***********************************************************************
  // ***********************************************************************
  //                             METHODS
  // ***********************************************************************
  /// Register field of structure.
  /// Note: fields are assumed to be stored in the same order they are pushed
  /// \tparam T base data type
  /// \param name field name
  /// \return field push id
  template<typename T>
  u64 pushField(const std::string &name) {
    field_id_map_[name] = fields_.size();
    Field d = {
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
  /// \param field_id field push id
  /// \return field name
  inline const std::string &fieldName(u64 field_id) const { return fields_[field_id].name; }
  u64 offsetOf(const std::string &field_name) const;
  u64 offsetOf(u64 field_id) const;
  u64 sizeOf(const std::string &field_name) const;
  u64 sizeOf(u64 field_id) const;
  inline u64 sizeInBytes() const { return struct_size_; }
  inline const std::vector<Field> &fields() const { return fields_; }
  /// \param o
  /// \param aos
  /// \return
  friend std::ostream &operator<<(std::ostream &o, const StructDescriptor &sd) {
    o << "Struct (size in bytes: " << sd.sizeInBytes() << ")\n";
    o << "fields: ";
    int i = 0;
    for (const auto &f : sd.fields_) {
      o << "field #" << i++ << " (" << f.name << " ): ";
      o << "\tbase data type: " << DataTypes::typeName(f.type) << "\n";
      o << "\tbase data size in bytes: " << f.size << "\n";
      o << "\tcomponent count: " << f.component_count << "\n";
    }
    return o;
  }

private:
  u64 struct_size_{0}; //!< in bytes
  std::vector<Field> fields_;
  std::unordered_map<std::string, u64> field_id_map_;
};

/// Provides access to a single struct field in a array of structs
/// \tparam T
template<typename T>
class AoSFieldAccessor {
public:
  AoSFieldAccessor(u8 *data, u64 stride, u64 offset) : data_{data}, stride_{stride}, offset_{offset} {}
  /// \param data
  void setDataPtr(u8 *data) { data_ = data; }
  /// \param i
  /// \return
  T &operator[](u64 i) {
    return *reinterpret_cast<T *>(data_ + i * stride_ + offset_);
  }

private:
  u8 *data_{nullptr};
  u64 stride_{0};
  u64 offset_{0};
};

/// Accesses a memory block as an array of structures
class AoSConstAccessor {
public:
  explicit AoSConstAccessor(const AoS &aos);
  /// \param descriptor
  explicit AoSConstAccessor(const StructDescriptor &descriptor) : struct_descriptor_{descriptor} {}
  /// \param descriptor
  /// \param data
  AoSConstAccessor(const StructDescriptor &descriptor, const u8 *data, u64 size) :
      struct_descriptor_{descriptor}, data_{data}, size_{size} {}
  ///
  /// \param data
  void setDataPtr(const u8 *data) { data_ = data; }
  /// \tparam T
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_ + i * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }

private:
  const StructDescriptor &struct_descriptor_;
  const u8 *data_{nullptr};
  u64 size_{0};
};

/// Accesses a memory block as an array of structures
class AoSAccessor {
public:
  explicit AoSAccessor(AoS &aos);
  explicit AoSAccessor(const StructDescriptor &descriptor) :
      struct_descriptor_{descriptor} {}
  /// \param descriptor
  /// \param data
  AoSAccessor(const StructDescriptor &descriptor, u8 *data, u64 size) :
      struct_descriptor_{descriptor}, data_{data}, size_{size} {}
  /// \param data
  void setDataPtr(u8 *data) { data_ = data; }
  /// \tparam T
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  const T &valueAt(u64 field_id, u64 i) const {
    return *reinterpret_cast<const T *>(data_ + i * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
  template<typename T>
  T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_ + i * struct_descriptor_.struct_size_
        + struct_descriptor_.fields_[field_id].offset);
  }
private:
  const StructDescriptor &struct_descriptor_;
  u8 *data_{nullptr};
  u64 size_{0};
};

/// Array of Structures
/// This class stores an array of structures that can be defined in runtime
class AoS {
  friend class AoSAccessor;
  friend class AoSConstAccessor;
public:
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
    if (!struct_descriptor.struct_size_) {
      spdlog::warn("[AoS] Fields must be previously registered.");
      return *this;
    }
    if (vector_data.size() * sizeof(T) % struct_descriptor.struct_size_ != 0)
      spdlog::warn("[AoS] Vector data with incompatible size.");
    delete[]data_;
    data_ = nullptr;
    size_ = vector_data.size() * sizeof(T) / struct_descriptor.struct_size_;
    data_ = new u8[size_ * struct_descriptor.struct_size_];
    std::memcpy(data_, vector_data.data(), size_ * struct_descriptor.struct_size_);
    return *this;
  }
  // ***********************************************************************
  //                             METHODS
  // ***********************************************************************
  const StructDescriptor &structDescriptor() const { return struct_descriptor; }
  /// \return
  [[nodiscard]] const u8 *data() const { return data_; }
  /// \param new_size in number of elements
  void resize(u64 new_size);

  AoSAccessor accessor() { return AoSAccessor(*this); }
  AoSConstAccessor constAccessor() const { return AoSConstAccessor(*this); }

  template<typename T>
  AoSFieldAccessor<T> field(const std::string &name) {
    auto it = struct_descriptor.field_id_map_.find(name);
    if (it == struct_descriptor.field_id_map_.end()) {
      spdlog::error("Field {} not found.", name);
      return AoSFieldAccessor<T>(nullptr, 0, 0);
    }
    return AoSFieldAccessor<T>(data_, struct_descriptor.struct_size_, struct_descriptor.fields_[it->second].offset);
  }
  template<typename T>
  u64 pushField(const std::string &name) {
    return struct_descriptor.pushField<T>(name);
  }
  /// \return
  [[nodiscard]] inline u64 size() const { return size_; }
  /// \return
  [[nodiscard]] inline u64 memorySizeInBytes() const { return size_ * struct_descriptor.struct_size_; }
  /// \return
  [[nodiscard]] inline u64 stride() const { return struct_descriptor.struct_size_; }

  /// \return
  const std::vector<StructDescriptor::Field> &fields() const { return struct_descriptor.fields_; }
  /// \tparam T
  /// \param field_id
  /// \param i
  /// \return
  template<typename T>
  T &valueAt(u64 field_id, u64 i) {
    return *reinterpret_cast<T *>(data_ + i * struct_descriptor.struct_size_
        + struct_descriptor.fields_[field_id].offset);
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
      << aos.struct_descriptor.sizeInBytes() << ")\n";
    o << "fields: ";
    int i = 0;
    for (const auto &f : aos.struct_descriptor.fields()) {
      o << "field #" << i++ << " (" << f.name << " ): ";
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
        offset += aos.struct_descriptor.sizeInBytes();
      }
      o << std::endl;
    }
    return o;
#undef PRINT_FIELD_VALUE
  }

private:
  u64 size_{0}; //!< struct count
  StructDescriptor struct_descriptor;
  u8 *data_{nullptr};
};

}

#endif //PONOS_PONOS_PONOS_STORAGE_ARRAY_OF_STRUCTURES_H
