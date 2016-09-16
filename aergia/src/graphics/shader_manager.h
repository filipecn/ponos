#ifndef AERGIA_GRAPHICS_SHADER_MANAGER_H
#define AERGIA_GRAPHICS_SHADER_MANAGER_H

#include "utils/open_gl.h"

#include <map>
#include<stdarg.h>
#include <string>
#include <vector>

namespace aergia {

	/* singleton
	 * Manages shader programs
	 */
	class ShaderManager {
		public:
			static ShaderManager &instance() {
				return instance_;
			}
			virtual ~ShaderManager() {}
			/* load
			 * Creates a shader program from shader files. It expects only
			 * one file of each type with extensions .fs, .vs and .gs.
			 * @return program id. **-1** if error occurred.
			 */
			int loadFromFiles(const char *fl...);
			/* use program
			 * @program **[in]** program's id
			 * Activate program
			 */
			void useShader(GLuint program);

		private:
			ShaderManager();
			ShaderManager(ShaderManager const&) = delete;
			void operator=(ShaderManager const&) = delete;

			GLuint createProgram(const GLchar*, const GLchar*);
			GLuint compile(const char* shaderSource, GLuint shaderType);
			GLuint createProgram(GLuint objects[], int size);

			static ShaderManager instance_;
	};

} // aergia namespace

#endif // AERGIA_GRAPHICS_SHADER_MANAGER_H

