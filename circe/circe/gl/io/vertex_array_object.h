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
///\file vertex_array_object.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-11-09
///
///\brief

#ifndef PONOS_CIRCE_CIRCE_GL_IO_VERTEX_ARRAY_OBJECT_H
#define PONOS_CIRCE_CIRCE_GL_IO_VERTEX_ARRAY_OBJECT_H

#include <circe/gl/io/buffer.h>

namespace circe::gl {

/// A Vertex Buffer Object (VBO) is the common term for a normal Buffer Object when
/// it is used as a source for vertex array data.
/// The VBO carries the description of attributes stored in the buffer. This
/// description is then used to pass the correct data to shaders.
///
/// Each vertex attribute data can be in a different buffer, or region of the same
/// buffer. Usually attributes are all stored in a single buffer, in interleaved
/// fashion.
/// Example:
/// Consider the following interleaved vertex buffer with N+1 vertices
/// px0 py0 pz0 u0 v0 nx0 ny0 nz0 ... pxN pyN pzN uN vN nxN nyN nzN
/// containing position, texture coordinates and normal for each vertex.
/// Data is stored in floats (4 bytes)
/// Note: Attributes must be pushed in order: position, texture coord. and normal.
/// The position attribute has offset 0 * 4, size 3 and type GL_FLOAT
/// The tex coord attribute has offset 3 * 4, size 2 and type GL_FLOAT
/// The normal attribute has offset 5 * 4, size 3 and type GL_FLOAT
class VertexBufferObject {
public:
  /// Attribute description
  /// Example:
  /// Consider a position attribute (ponos::point3, f32 p[3], ...). Its description
  /// would be
  ///  - name = "position" (optional)
  ///  - size = 3 (x y z)
  ///  - type = GL_FLOAT
  ///  - binding_point = depends on the buffer setup (usually is 0)
  struct Attribute {
    std::string name; //!< attribute name (optional)
    u64 size{0};    //!< attribute number of components
    GLenum type{0}; //!< attribute data type (GL_FLOAT,...)
    GLuint binding_index{0}; //!< buffer binding point (0 to GL_MAX_VERTEX_ATTRIB_BINDINGS - 1)
  };
  /// Default constructor
  VertexBufferObject();
  /// Param constructor
  /// \param attributes
  explicit VertexBufferObject(const std::vector<Attribute> &attributes);
  ~VertexBufferObject();
  /// \param attributes
  void pushAttributes(const std::vector<Attribute> &attributes);
  /// Append attribute to vertex format description. Push order is important, as
  /// attribute offsets are computed as attributes are appended.
  /// \param component_count
  /// \param name attribute name
  /// \param data_type
  /// \param binding_index
  /// \return attribute id
  u64 pushAttribute(u64 component_count, const std::string &name = "", GLenum data_type = GL_FLOAT,
                    GLuint binding_index = 0);
  /// Push attribute using a ponos type
  /// \tparam T accepts only ponos::MathElement derived objects (point, vector, matrix,...)
  /// \param name attribute name (optional)
  /// \return attribute id
  template<typename T,
      typename std::enable_if_t<
          std::is_base_of_v<ponos::MathElement<f32, 2u>, T> ||
              std::is_base_of_v<ponos::MathElement<f32, 3u>, T> ||
              std::is_base_of_v<ponos::MathElement<f32, 4u>, T> ||
              std::is_base_of_v<ponos::MathElement<f32, 9u>, T> ||
              std::is_base_of_v<ponos::MathElement<f32, 16u>, T>> * = nullptr>
  u64 pushAttribute(const std::string &name = "") {
    return pushAttribute(T::componentCount(), name, OpenGL::dataTypeEnum<decltype(T::numeric_data)>());
  }
  /// Note: A vertex array object must be bound before calling this method
  void specifyVertexFormat();

private:
  void updateOffsets();

  std::unordered_map<std::string, u64> attribute_name_id_map_;
  std::vector<Attribute> attributes_;
  std::vector<u64> offsets_;  //!< attribute data offset (in bytes)
};
/// A Vertex Array Object (VAO) is an OpenGL Object that stores all of the
/// state needed to supply vertex data. It stores the format of the vertex data
/// as well as the Buffer Objects (see below) providing the vertex data arrays.
class VertexArrayObject {
public:
private:
};

}

#endif //PONOS_CIRCE_CIRCE_GL_IO_VERTEX_ARRAY_OBJECT_H
