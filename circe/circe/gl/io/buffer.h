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

#ifndef CIRCE_IO_BUFFER_H
#define CIRCE_IO_BUFFER_H

#include <circe/gl/utils/open_gl.h>

#include <ponos/ponos.h>

#include <map>
#include <string>
#include <utility>

namespace circe::gl {

/// Describes the data contained in a buffer.
struct BufferDescriptor {
  struct Attribute {
    u32 offset = 0;  //!< attribute data offset (in bytes)
    u32 size = 0;    //!< attribute number of components
    GLenum type = 0; //!< attribute data type
  };
  std::map<const std::string, Attribute> attributes; //!< name - attribute map
  GLuint element_size = 0; //!< how many components are assigned to each element
  u32 element_count = 0;   //!< number of elements
  GLuint element_type;     //!< type of elements  (GL_TRIANGLES, ...)
  GLuint type;            //!< buffer type (GL_ARRAY_BUFFER, ...)
  GLuint use;             //!< use  (GL_STATIC_DRAW, ...)
  GLuint data_type;        //!< (GL_FLOAT, ...)
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      CONSTRUCTORS     //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// Default constructor
  BufferDescriptor()
      : element_size(0), element_count(0), element_type(GL_TRIANGLES),
        type(GL_ARRAY_BUFFER), use(GL_STATIC_DRAW), data_type(GL_FLOAT) {}
  /// \brief Constructor.
  /// \param _elementSize **[in]** number of data components that compose an
  /// element
  /// \param _elementCount **[in]** number of elements
  /// \param _elementType **[in | optional]** element type
  /// \param _type **[in | optional]** buffer type
  /// \param _use **[in | optional]** buffer use
  /// \param _dataType **[in | optional]** data type
  BufferDescriptor(GLuint _elementSize, u32 _elementCount,
                   GLuint _elementType = GL_TRIANGLES,
                   GLuint _type = GL_ARRAY_BUFFER, GLuint _use = GL_STATIC_DRAW,
                   GLuint _dataType = GL_FLOAT)
      : element_size(_elementSize), element_count(_elementCount),
        element_type(_elementType), type(_type), use(_use), data_type(_dataType) {
  }
  ///
  /// \param other
  BufferDescriptor(const BufferDescriptor &other) {
    element_size = other.element_size;
    element_count = other.element_count;
    element_type = other.element_type;
    type = other.type;
    use = other.use;
    data_type = other.data_type;
    for (auto a : other.attributes)
      attributes[a.first] = a.second;
  }
  ///
  /// \param other
  BufferDescriptor(const BufferDescriptor &&other) {
    element_size = other.element_size;
    element_count = other.element_count;
    element_type = other.element_type;
    type = other.type;
    use = other.use;
    data_type = other.data_type;
    for (auto a : other.attributes)
      attributes[a.first] = a.second;
  }
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      OPERATORS        //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  BufferDescriptor &operator=(const BufferDescriptor &other) {
    element_size = other.element_size;
    element_count = other.element_count;
    element_type = other.element_type;
    type = other.type;
    use = other.use;
    data_type = other.data_type;
    for (auto a : other.attributes)
      attributes[a.first] = a.second;
    return *this;
  }
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////        METHODS        //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// \param _name attribute name (used in shader programs)
  /// \param _size number of components
  /// \param _offset attribute data offset (in bytes)
  /// \param type attribute data type (GL_FLOAT)
  void addAttribute(const std::string &_name, size_t _size, uint _offset,
                    GLenum _type) {
    Attribute att;
    att.size = _size;
    att.offset = _offset;
    att.type = _type;
    attributes[_name] = att;
  }
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////        STATIC         //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// \param elementSize
  /// \param dataType
  /// \return BufferDescriptor for an array stream
  static BufferDescriptor forArrayStream(GLuint elementSize, GLuint dataType = GL_FLOAT) {
    return BufferDescriptor(elementSize, 0, GL_TRIANGLES, GL_ARRAY_BUFFER,
                            GL_STREAM_DRAW, dataType);
  }
  /// \param elementSize
  /// \param elementCount
  /// \param elementType
  /// \return BufferDescriptor for a vertex buffer
  static BufferDescriptor forVertexBuffer(size_t elementSize, size_t elementCount,
                                          GLuint elementType = GL_TRIANGLES) {
    return BufferDescriptor(elementSize, elementCount, elementType,
                            GL_ARRAY_BUFFER);
  }
  ///
  /// \param elementSize
  /// \param elementCount
  /// \param elementType
  /// \return BufferDescriptor for an index buffer
  static BufferDescriptor forIndexBuffer(size_t elementSize, size_t elementCount,
                                         ponos::GeometricPrimitiveType elementType) {
    GLuint type = GL_TRIANGLES;
    switch (elementType) {
    case ponos::GeometricPrimitiveType::TRIANGLES:type = GL_TRIANGLES;
      break;
    case ponos::GeometricPrimitiveType::LINES:type = GL_LINES;
      break;
    case ponos::GeometricPrimitiveType::QUADS:type = GL_QUADS;
      break;
    case ponos::GeometricPrimitiveType::LINE_LOOP:type = GL_LINE_LOOP;
      break;
    case ponos::GeometricPrimitiveType::TRIANGLE_FAN:type = GL_TRIANGLE_FAN;
      break;
    case ponos::GeometricPrimitiveType::TRIANGLE_STRIP:type = GL_TRIANGLE_STRIP;
      break;
    default:break;
    }
    return BufferDescriptor(elementSize, elementCount, type,
                            GL_ELEMENT_ARRAY_BUFFER);
  }
};

/// \brief Set the up buffer data from mesh object
/// Interleaves vertex, normal and uv data into a single buffer (vertexData)
/// handling the indices (duplicating data if needed)
/// \param rm
/// \param vertexData
/// \param indexData
inline void setup_buffer_data_from_mesh(const ponos::RawMesh &rm,
                                        std::vector<float> &vertexData,
                                        std::vector<uint> &indexData) {
  // convert mesh to buffers
  // position index | norm index | texcoord index -> index
  typedef std::pair<std::pair<int, int>, int> MapKey;
  std::map<MapKey, size_t> m;
  size_t newIndex = 0;
  for (auto &i : rm.indices) {
    auto key = MapKey(std::pair<int, int>(i.positionIndex, i.normalIndex),
                      i.texcoordIndex);
    auto it = m.find(key);
    auto index = newIndex;
    if (it == m.end()) {
      // add data
      if (rm.positionDescriptor.count)
        for (size_t v = 0; v < rm.positionDescriptor.elementSize; v++)
          vertexData.emplace_back(
              rm.positions[i.positionIndex * rm.positionDescriptor.elementSize +
                  v]);
      if (rm.normalDescriptor.count)
        for (size_t n = 0; n < rm.normalDescriptor.elementSize; n++)
          vertexData.emplace_back(
              rm.normals[i.normalIndex * rm.normalDescriptor.elementSize + n]);
      if (rm.texcoordDescriptor.count)
        for (size_t t = 0; t < rm.texcoordDescriptor.elementSize; t++)
          vertexData.emplace_back(
              rm.texcoords[i.texcoordIndex * rm.texcoordDescriptor.elementSize +
                  t]);
      m[key] = newIndex++;
    } else
      index = it->second;
    indexData.emplace_back(index);
  }
}

/// Creates buffer descriptions from a raw mesh interleaved data
/// \param m **[in]** raw mesh (interleaved data must be built previously)
/// \param v **[out]** vertex buffer description
/// \param i **[out]** index buffer description
inline void create_buffer_description_from_mesh(const ponos::RawMesh &m,
                                                BufferDescriptor &v,
                                                BufferDescriptor &i,
                                                GLuint use = GL_STATIC_DRAW) {
  GLuint type = GL_TRIANGLES;
  switch (m.primitiveType) {
  case ponos::GeometricPrimitiveType::TRIANGLES:type = GL_TRIANGLES;
    break;
  case ponos::GeometricPrimitiveType::LINES:type = GL_LINES;
    break;
  case ponos::GeometricPrimitiveType::QUADS:type = GL_QUADS;
    break;
  case ponos::GeometricPrimitiveType::LINE_LOOP:type = GL_LINE_LOOP;
    break;
  case ponos::GeometricPrimitiveType::TRIANGLE_FAN:type = GL_TRIANGLE_FAN;
    break;
  case ponos::GeometricPrimitiveType::TRIANGLE_STRIP:type = GL_TRIANGLE_STRIP;
    break;
  default:break;
  }
  i.element_type = type;
  i.element_count = m.meshDescriptor.count;
  i.element_size = m.meshDescriptor.elementSize;
  i.type = GL_ELEMENT_ARRAY_BUFFER;
  i.use = use;
  i.data_type = GL_UNSIGNED_INT;
  v.element_count = m.interleavedDescriptor.count;
  v.element_size = m.interleavedDescriptor.elementSize;
  v.type = GL_ARRAY_BUFFER;
  v.use = use;
  v.data_type = GL_FLOAT;
  v.addAttribute(std::string("position"), m.positionDescriptor.elementSize, 0,
                 GL_FLOAT);
  size_t offset = m.positionDescriptor.elementSize;
  if (m.normalDescriptor.count) {
    v.addAttribute(std::string("normal"), m.normalDescriptor.elementSize,
                   offset * sizeof(float), GL_FLOAT);
    offset += m.normalDescriptor.elementSize;
  }
  if (m.texcoordDescriptor.count) {
    v.addAttribute(std::string("texcoord"), m.texcoordDescriptor.elementSize,
                   offset * sizeof(float), GL_FLOAT);
    offset += m.texcoordDescriptor.elementSize;
  }
}

/// forward declaration of ShaderProgram
class ShaderProgram;
class Program;

/// Interface for device buffers accessed by a shader.
/// A device buffer used by a shader must have its data laid out in memory in a
/// way compatible to the shader layout. It is done by mapping the shader
/// attribute locations to the buffer.
class BufferInterface {
public:
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      CONSTRUCTORS     //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// Default constructor
  explicit BufferInterface(GLuint id = 0);
  ///
  /// \param b
  /// \param id
  explicit BufferInterface(BufferDescriptor b, GLuint id = 0);
  /// Destructor
  virtual ~BufferInterface();
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      METHODS          //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// Activate buffer
  void bind() const;
  /// \brief register attribute
  /// \param name **[in]** attribute name
  /// \param location **[in]** attribute location on shader program
  void registerAttribute(const std::string &name, GLint location) const;
  /// \return open gl's buffer id
  GLuint id() const;
  /// locates and register buffer attributes in shader program
  /// \param s shader
  /// \param d **[optional]** attribute divisor (default = 0)
  void locateAttributes(const ShaderProgram &s, uint d = 0) const;
  void locateAttributes(const Program &s, uint d = 0) const;
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      FIELDS           //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  BufferDescriptor bufferDescriptor; //!< buffer description

protected:
  GLuint bufferId;
};

/// Device buffer for shader access
/// \tparam T data type
template<typename T>
class GLBuffer : public BufferInterface {
public:
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////      CONSTRUCTORS     //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  [[deprecated]]
  GLBuffer() = default;
  /// \brief Constructor
  /// \param d **[in]** data pointer
  /// \param bd **[in]** buffer description
  [[deprecated]]
  GLBuffer(const T *d, const BufferDescriptor &bd) : GLBuffer() { set(d, bd); }
  ///  \brief Constructor
  /// \param id **[in]** an existent buffer
  /// \param bd **[in]** buffer description
  [[deprecated]]
  explicit GLBuffer(const BufferDescriptor &bd, GLuint id = 0)
      : BufferInterface(bd, id) {}

  ~GLBuffer() override {
    // already destroying GL buffer in base destructor!
    // glDeleteBuffers(1, &this->bufferId);
  }
  /////////////////////////////////////////////////////////////////////////////
  ////////////////////////         METHODS       //////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  /// Changes size of buffer
  /// Note: While creating the new storage, any pre-existing data store is
  /// deleted
  /// \param size new size (in number of elements)
  void resize(int size) {
    this->bufferDescriptor.element_count = size;
    if (!this->bufferId)
      return;
    glBindBuffer(this->bufferDescriptor.type, this->bufferId);
    glBufferData(this->bufferDescriptor.type,
                 this->bufferDescriptor.element_count *
                     this->bufferDescriptor.element_size * sizeof(T),
                 nullptr, this->bufferDescriptor.use);
    CHECK_GL_ERRORS;
  }
  /// \brief sets data and description
  /// Notes:
  ///     - Previous buffer object is destroyed
  ///     - The buffer remains bound
  /// \param data **[in]** data pointer
  /// \param bd **[in]** buffer description
  void set(const T *data, const BufferDescriptor &bd) {
    this->bufferDescriptor = bd;
    if (this->bufferId > 0)
      glDeleteBuffers(1, &this->bufferId);
    glGenBuffers(1, &this->bufferId);
    glBindBuffer(this->bufferDescriptor.type, this->bufferId);
    glBufferData(this->bufferDescriptor.type,
                 this->bufferDescriptor.element_count *
                     this->bufferDescriptor.element_size * sizeof(T),
                 data, this->bufferDescriptor.use);
    CHECK_GL_ERRORS;
  }
  /// Overwrites data in buffer's memory.
  /// Note: it assumes same buffer size from its description
  /// \param data **[in]** data pointer
  void set(const T *data) {
    if (this->bufferId == 0) {
      set(data, this->bufferDescriptor);
      CHECK_GL_ERRORS;
      return;
    }
    glBindBuffer(this->bufferDescriptor.type, this->bufferId);
    glBufferSubData(this->bufferDescriptor.type, 0,
                    this->bufferDescriptor.element_count *
                        this->bufferDescriptor.element_size * sizeof(T),
                    data);
    CHECK_GL_ERRORS;
  }
};

using VertexBuffer = GLBuffer<float>;
using IndexBuffer = GLBuffer<uint>;

/// Holds an Open GL Buffer Object that store an array of unformatted memory allocated
/// by the GPU. These can be used to store vertex data, pixel data retrieved from
/// images or the framebuffer, and a variety of other things.
///
/// Buffer Targets:
/// GL_ARRAY_BUFFER, GL_ATOMIC_COUNTER_BUFFER, GL_COPY_READ_BUFFER,
/// GL_COPY_WRITE_BUFFER, GL_DRAW_INDIRECT_BUFFER, GL_DISPATCH_INDIRECT_BUFFER,
/// GL_ELEMENT_ARRAY_BUFFER, GL_PIXEL_PACK_BUFFER, GL_PIXEL_UNPACK_BUFFER,
/// GL_QUERY_BUFFER, GL_SHADER_STORAGE_BUFFER, GL_TEXTURE_BUFFER,
/// GL_TRANSFORM_FEEDBACK_BUFFER, or GL_UNIFORM_BUFFER.
///
/// Buffer Usages:
/// GL_STREAM_DRAW, GL_STREAM_READ, GL_STREAM_COPY, GL_STATIC_DRAW, GL_STATIC_READ,
/// GL_STATIC_COPY, GL_DYNAMIC_DRAW, GL_DYNAMIC_READ, or GL_DYNAMIC_COPY
///
/// Notes:
/// - This class uses RAII. The object is created on construction and destroyed on
/// deletion.
class Buffer {
public:
  Buffer();
  /// Parameters constructor
  /// \param usage Specifies the expected usage pattern of the data store.
  /// (ex: GL_STATIC_DRAW)
  /// \param target Specifies the target buffer object. (ex: GL_ARRAY_BUFFER)
  /// \param size Specifies the size in bytes of the buffer object.
  /// \param data Specifies a pointer to data that will be copied into the data
  /// store for initialization.
  Buffer(GLuint usage, GLuint target, u64 size = 0, void* data = nullptr);
  virtual ~Buffer();
  /// \param _target Specifies the target buffer object. (ex: GL_ARRAY_BUFFER)
  void setTarget(GLuint _target);
  /// \param usage Specifies the expected usage pattern of the data store.
  /// (ex: GL_STATIC_DRAW)
  void setUsage(GLuint _usage);
  /// \param size Specifies the size in bytes of the buffer object.
  void resize(u64 size_in_bytes);
  /// \return buffer size in bytes
  [[nodiscard]] inline u64 size() const { return size_; }
  /// \return buffer usage
  [[nodiscard]] inline GLuint usage() const { return usage_; }
  /// \return buffer target
  [[nodiscard]] inline GLuint target() const { return target_; }
  /// Deletes buffer data and destroy object
  void destroy();
private:
  GLuint buffer_object_id_{0};
  u64 size_{0};
  GLuint target_{0};            //!< buffer type (GL_ARRAY_BUFFER, ...)
  GLuint usage_{0};             //!< use  (GL_STATIC_DRAW, ...)
};

} // namespace circe

#endif // CIRCE_IO_BUFFER_H
