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

AoS::AoS() = default;

AoS::~AoS() {
  delete[]data_;
}

AoS &AoS::operator=(AoS &&other) noexcept {
  delete[]data_;
  data_ = other.data_;
  other.data_ = nullptr;
  size_ = other.size_;
  struct_size_ = other.struct_size_;
  fields_ = std::move(other.fields_);
  field_id_map_ = std::move(other.field_id_map_);
  return *this;
}

AoS &AoS::operator=(const AoS &other) {
  delete[]data_;
  data_ = nullptr;
  size_ = other.size_;
  struct_size_ = other.struct_size_;
  fields_ = other.fields_;
  field_id_map_ = other.field_id_map_;
  if (size_ && struct_size_) {
    data_ = new u8[size_ * struct_size_];
    std::memcpy(data_, other.data_, size_ * struct_size_);
  }
  return *this;
}

void AoS::resize(u64 new_size) {
  delete[]data_;
  data_ = nullptr;
  size_ = new_size;
  if (!new_size)
    return;
  data_ = new u8[new_size * struct_size_];
}

u64 AoS::offsetOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    spdlog::error("Field {} not found.", field_name);
    return 0;
  }
  return offsetOf(it->second);
}

u64 AoS::offsetOf(u64 field_id) const {
  return fields_[field_id].offset;
}

u64 AoS::sizeOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    spdlog::error("Field {} not found.", field_name);
    return 0;
  }
  return sizeOf(it->second);
}
u64 AoS::sizeOf(u64 field_id) const {
  return fields_[field_id].size;
}

}
