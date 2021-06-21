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
///\file array_of_structures.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-12-10
///
///\brief


#include <ponos/storage/array_of_structures.h>

namespace ponos {

u64 StructDescriptor::offsetOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    Log::error("Field {} not found.", field_name);
    return 0;
  }
  return offsetOf(it->second);
}

u64 StructDescriptor::offsetOf(u64 field_id) const {
  return fields_[field_id].offset;
}

u64 StructDescriptor::sizeOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    Log::error("Field {} not found.", field_name);
    return 0;
  }
  return sizeOf(it->second);
}

u64 StructDescriptor::sizeOf(u64 field_id) const {
  return fields_[field_id].size;
}

AoS::ConstAccessor::ConstAccessor(const AoS &aos) : struct_descriptor_{aos.structDescriptor()}, data_{aos.data_},
                                                     size_{aos.size_} {}

AoS::Accessor::Accessor(AoS &aos) : struct_descriptor_{aos.structDescriptor()}, data_{aos.data_},
                                     size_{aos.size_} {}

AoS::AoS() = default;

AoS::~AoS() {
  delete[]data_;
}

AoS &AoS::operator=(AoS &&other) noexcept {
  delete[]data_;
  data_ = other.data_;
  other.data_ = nullptr;
  size_ = other.size_;
  struct_descriptor.struct_size_ = other.struct_descriptor.struct_size_;
  struct_descriptor.fields_ = std::move(other.struct_descriptor.fields_);
  struct_descriptor.field_id_map_ = std::move(other.struct_descriptor.field_id_map_);
  return *this;
}

AoS &AoS::operator=(const AoS &other) {
  delete[]data_;
  data_ = nullptr;
  size_ = other.size_;
  struct_descriptor.struct_size_ = other.struct_descriptor.struct_size_;
  struct_descriptor.fields_ = other.struct_descriptor.fields_;
  struct_descriptor.field_id_map_ = other.struct_descriptor.field_id_map_;
  if (size_ && struct_descriptor.struct_size_) {
    data_ = new u8[size_ * struct_descriptor.struct_size_];
    std::memcpy(data_, other.data_, size_ * struct_descriptor.struct_size_);
  }
  return *this;
}

void AoS::resize(u64 new_size) {
  delete[]data_;
  data_ = nullptr;
  size_ = new_size;
  if (!new_size)
    return;
  data_ = new u8[new_size * struct_descriptor.struct_size_];
}

}
