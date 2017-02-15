#ifndef AERGIA_GRAPHICS_SHADER_H
#define AERGIA_GRAPHICS_SHADER_H

#include "graphics/shader_manager.h"

#include <ponos.h>

#include <vector>

namespace aergia {

	/* shader class
	 * Holds a program id and serves as an interface for setting its uniforms.
	 */
	class Shader {
		public:
			Shader(GLuint id = 0);
			/* load
			 * Creates a shader program from shader files. It expects only
			 * one file of each type with extensions .fs, .vs and .gs.
			 * @return program id. **-1** if error.
			 */
			bool loadFromFiles(const char *fl...);
			/* begin
			 * Acctivate shader program
			 */
			bool begin();
			/* end
			 * Deactivate shader program
			 */
			void end();

			int vDataSize;
			void addVertexAttribute(const char* attribute, int size, int offset);
			// Uniforms
			void setUniform(const char* name, const ponos::mat4 &m);
			void setUniform(const char* name, const ponos::mat3 &m);
			void setUniform(const char* name, const ponos::vec4 &v);
			void setUniform(const char* name, const ponos::vec3 &v);
			void setUniform(const char* name, const ponos::vec2 &v);
			void setUniform(const char* name, int i);
			void setUniform(const char* name, float f);

			bool running;

		protected:
			GLuint programId;

			struct VertexAttribute{
				const char* name;
				int offset;
				int size;
			};

			std::vector<VertexAttribute> vertexAttributes;

			GLint getUniLoc(const GLchar *name);
	};

} // aergia namespace

#endif // AERGIA_GRAPHICS_SHADER_H

