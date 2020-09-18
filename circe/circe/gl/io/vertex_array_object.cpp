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
///\file vertex_array_object.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-11-09
///
///\brief

#include "vertex_array_object.h"

namespace circe::gl {

VertexBufferObject::VertexBufferObject() = default;

VertexBufferObject::VertexBufferObject(const std::vector<Attribute> &attributes) {
  pushAttributes(attributes);
}

VertexBufferObject::~VertexBufferObject() = default;

void VertexBufferObject::pushAttributes(const std::vector<Attribute> &attributes) {
  attributes_ = attributes;
  updateOffsets();
}

u64 VertexBufferObject::pushAttribute(u64 component_count, const std::string &name, GLenum data_type,
                                      GLuint binding_point) {
  u64 attribute_id = attributes_.size();
  std::string attribute_name = name.empty() ? std::to_string(attribute_id) : name;
  Attribute attr{
      attribute_name,
      component_count,
      data_type,
      binding_point
  };
  attributes_.emplace_back(attr);
  attribute_name_id_map_[attribute_name] = attribute_id;
  return attribute_id;
}

void VertexBufferObject::updateOffsets() {
  if (attributes_.empty()) {
    offsets_.clear();
    return;
  }
  offsets_.resize(attributes_.size());
  u64 offset = 0;
  for (const auto &a : attributes_) {
    offsets_.emplace_back(offset);
    offset += a.size * OpenGL::dataSizeInBytes(a.type);
  }
}

void VertexBufferObject::specifyVertexFormat() {
  for (u64 i = 0; i < attributes_.size(); ++i) {
    // specify vertex attribute format
    glVertexAttribFormat(i, attributes_[i].size, attributes_[i].type, false, offsets_[i]);
    // set the details of a single attribute
    glVertexAttribBinding(i, attributes_[i].binding_index);
    // which buffer binding point it is attached to
    glEnableVertexAttribArray(i);
  }
}

}
