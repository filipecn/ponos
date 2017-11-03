/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
*/

#ifndef AERGIA_IO_BUFFER_H
#define AERGIA_IO_BUFFER_H

#include <aergia/utils/open_gl.h>

#include <ponos/ponos.h>

#include <map>
#include <string>

namespace aergia {

/** Describes the data contained in a buffer.
 */
struct BufferDescriptor {
  struct Attribute {
    long offset; //!< attribute data offset (in bytes)
    size_t size; //!< attribute number of components
    GLenum type; //!< attribute data type
  };
  std::map<const std::string, Attribute> attributes; //!< name - attribute map
  GLuint elementSize;  //!< how many components are assigned to each element
  size_t elementCount; //!< number of elements
  GLuint elementType;  //!< type of elements
  GLuint type;         //!< buffer type
  GLuint use;          //!< use
  BufferDescriptor() {}
  /** \brief Constructor.
   * \param _elementSize **[in]** number of data components that compose an
   * element
   * \param _elementCount **[in]** number of elements
   * \param _elementType **[in | optional]** element type
   * \param _type **[in | optional]** buffer type
   * \param _use **[in | optional]** buffer use
   */
  BufferDescriptor(GLuint _elementSize, size_t _elementCount,
                   GLuint _elementType = GL_TRIANGLES,
                   GLuint _type = GL_ARRAY_BUFFER, GLuint _use = GL_STATIC_DRAW)
      : elementSize(_elementSize), elementCount(_elementCount),
        elementType(_elementType), type(_type), use(_use) {}
  /** \brief  add
   * \param _name attribute name (used in shader programs)
   * \param _size number of components
   * \param _offset attribute data offset (in bytes)
   * \param type attribute data type
   */
  void addAttribute(const std::string &_name, size_t _size, uint _offset,
                    GLenum _type) {
    Attribute att;
    att.size = _size;
    att.offset = _offset;
    att.type = _type;
    attributes[_name] = att;
  }
};

inline BufferDescriptor
create_vertex_buffer_descriptor(size_t elementSize, size_t elementCount,
                                GLuint elementType = GL_TRIANGLES) {
  return BufferDescriptor(elementSize, elementCount, elementType,
                          GL_ARRAY_BUFFER);
}

inline BufferDescriptor
create_index_buffer_descriptor(size_t elementSize, size_t elementCount,
                               ponos::GeometricPrimitiveType elementType) {
  GLuint type = GL_TRIANGLES;
  switch (elementType) {
  case ponos::GeometricPrimitiveType::TRIANGLES:
    type = GL_TRIANGLES;
    break;
  case ponos::GeometricPrimitiveType::LINES:
    type = GL_LINES;
    break;
  case ponos::GeometricPrimitiveType::QUADS:
    type = GL_QUADS;
    break;
  case ponos::GeometricPrimitiveType::LINE_LOOP:
    type = GL_LINE_LOOP;
    break;
  case ponos::GeometricPrimitiveType::TRIANGLE_FAN:
    type = GL_TRIANGLE_FAN;
    break;
  case ponos::GeometricPrimitiveType::TRIANGLE_STRIP:
    type = GL_TRIANGLE_STRIP;
    break;
  default:
    break;
  }
  return BufferDescriptor(elementSize, elementCount, type,
                          GL_ELEMENT_ARRAY_BUFFER);
}

template <typename T> class Buffer {
public:
  Buffer() {}
  /** \brief Constructor
   * \param d **[in]** data pointer
   * \param bd **[in]** buffer description
   */
  Buffer(const T *d, const BufferDescriptor &bd) : Buffer() { set(d, bd); }
  virtual ~Buffer() { glDeleteBuffers(1, &bufferId); }
  /** \brief set
   * \param d **[in]** data pointer
   * \param bd **[in]** buffer description
   */
  void set(const T *d, const BufferDescriptor &bd) {
    data = d;
    bufferDescriptor = bd;
    glGenBuffers(1, &bufferId);
    glBindBuffer(bufferDescriptor.type, bufferId);
    glBufferData(bufferDescriptor.type,
                 bufferDescriptor.elementCount * bufferDescriptor.elementSize *
                     sizeof(T),
                 data, bufferDescriptor.use);
  }
  /** \brief set
   * \param d **[in]** data pointer
   */
  void set(const T *d) {
    data = d;
    glBindBuffer(bufferDescriptor.type, bufferId);
    glBufferSubData(bufferDescriptor.type, 0,
                    bufferDescriptor.elementCount *
                        bufferDescriptor.elementSize * sizeof(T),
                    data);
  }
  /** \brief Activate buffer
   */
  void bind() const { glBindBuffer(bufferDescriptor.type, bufferId); }
  /* \brief register attribute
   * \param name **[in]** attribute name
   * \param location **[in]** attribute location on shader program
   */
  void registerAttribute(const std::string &name, GLint location) const {
    if (bufferDescriptor.attributes.find(name) ==
        bufferDescriptor.attributes.end())
      return;
    const BufferDescriptor::Attribute &va =
        bufferDescriptor.attributes.find(name)->second;
    glVertexAttribPointer(location, va.size, va.type, GL_FALSE,
                          bufferDescriptor.elementSize * sizeof(float),
                          (void *)(va.offset));
  }

  BufferDescriptor bufferDescriptor; //!< buffer description

protected:
  GLuint bufferId;
  const T *data;
};

typedef Buffer<float> VertexBuffer;
typedef Buffer<uint> IndexBuffer;

} // aergia namespace

#endif // AERGIA_IO_BUFFER_H
