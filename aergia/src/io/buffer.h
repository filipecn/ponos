#ifndef AERGIA_IO_BUFFER_H
#define AERGIA_IO_BUFFER_H

#include "utils/open_gl.h"

#include <ponos.h>

#include <string>
#include <map>

namespace aergia {

/* descriptor
 * Describes the data contained in a buffer.
 */
struct BufferDescriptor {
  struct Attribute {
    // attribute data offset (in bytes)
    uint offset;
    // attribute data size (in bytes)
    size_t size;
    // attribute data type
    GLenum type;
  };
  // name - attribute map
  std::map<std::string, Attribute> attributes;
  // how many components are assigned to each element
  GLuint elementSize;
  // number of elements
  size_t elementCount;
  // type of elements
  GLuint elementType;
  // buffer type
  GLuint type;
  // use
  GLuint use;
  BufferDescriptor() {}
  /* Constructor.
   * @_elementSize **[in]** number of data components compose an element
   * @_elementCount **[in]** number of elements
   * @_elementType **[in | optional]** element type
   * @_type **[in | optional]** buffer type
   * @_use **[in | optional]** buffer use
   */
  BufferDescriptor(GLuint _elementSize, size_t _elementCount,
                   GLuint _elementType = GL_TRIANGLES,
                   GLuint _type = GL_ARRAY_BUFFER, GLuint _use = GL_STATIC_DRAW)
      : elementSize(_elementSize), elementCount(_elementCount),
        elementType(_elementType), type(_type), use(_use) {}
};

inline BufferDescriptor
create_vertex_buffer_descriptor(size_t elementSize, size_t elementCount,
                                GLuint elementType = GL_TRIANGLES) {
  return BufferDescriptor(elementSize, elementCount, elementType,
                          GL_ARRAY_BUFFER);
}

inline BufferDescriptor
create_index_buffer_descriptor(size_t elementSize, size_t elementCount,
                               GLuint elementType = GL_TRIANGLES) {
  return BufferDescriptor(elementSize, elementCount, elementType,
                          GL_ELEMENT_ARRAY_BUFFER);
}

template <typename T> class Buffer {
public:
  Buffer() {}
  /* Constructor
   * @d **[in]** data pointer
   * @bd **[in]** buffer description
   */
  Buffer(const T *d, const BufferDescriptor &bd) : Buffer() { set(d, bd); }
  virtual ~Buffer() { glDeleteBuffers(1, &bufferId); }
  /* set
   * @d **[in]** data pointer
   * @bd **[in]** buffer description
   */
  void set(const T *d, const BufferDescriptor &bd) {
    data.reset(d);
    bufferDescriptor = bd;
    glGenBuffers(1, &bufferId);
    glBindBuffer(bufferDescriptor.type, bufferId);
    glBufferData(bufferDescriptor.type,
                 bufferDescriptor.elementCount * bufferDescriptor.elementSize *
                     sizeof(T),
                 data.get(), bufferDescriptor.use);
  }
  /* bind
   * Activate buffer
   */
  void bind() const { glBindBuffer(bufferDescriptor.type, bufferId); }
  /* register
   * @name **[in]** attribute name
   * @location **[in]** attribute location on shader program
   */
  void registerAttribute(const std::string &name, GLint location) {
    if (bufferDescriptor.attributes.find(name) ==
        bufferDescriptor.attributes.end())
      return;
    const BufferDescriptor::Attribute &va = bufferDescriptor.attributes[name];
    glVertexAttribPointer(location, bufferDescriptor.elementSize, va.type,
                          GL_FALSE, va.size, (void *)(va.offset));
  }
  // buffer description
  BufferDescriptor bufferDescriptor;

protected:
  GLuint bufferId;
  std::shared_ptr<const T> data;
};

typedef Buffer<float> VertexBuffer;
typedef Buffer<uint> IndexBuffer;

} // aergia namespace

#endif // AERGIA_IO_BUFFER_H
