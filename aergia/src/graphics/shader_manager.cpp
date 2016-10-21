#include "graphics/shader_manager.h"

namespace aergia {

	ShaderManager ShaderManager::instance_;

	ShaderManager::ShaderManager() {}

	int ShaderManager::loadFromFiles(const char *fl...) {
		va_list args;
		va_start(args, fl);
		GLuint objects[] = {0, 0, 0};
		GLuint types[] = {GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER};
		while(*fl != '\0') {
			std::string filename(fl);
			if(filename.size() < 4)
				continue;
			char *source = nullptr;
			if(!ponos::readFile(filename.c_str(), &source))
				continue;
			GLuint shaderType = 0;
			switch(filename[filename.size() - 2]) {
				case 'v': shaderType = 0; break;
				case 'g': shaderType = 1; break;
				case 'f': shaderType = 2; break;
				default: continue;
			}
			objects[shaderType] = compile(source, types[shaderType]);
			if(source)
				free(source);
			++fl;
		}
		va_end(args);
		GLuint program = createProgram(objects, 3);
		if(!program)
			return -1;
		return static_cast<int>(program);
	}

	int ShaderManager::loadFromTexts(const char *vs, const char *gs, const char *fs) {
		GLuint objects[] = {0, 0, 0};
		if(vs != nullptr)
			objects[0] = compile(vs, GL_VERTEX_SHADER);
		if(gs != nullptr)
			objects[1] = compile(gs, GL_GEOMETRY_SHADER);
		if(fs != nullptr)
			objects[2] = compile(fs, GL_FRAGMENT_SHADER);
		GLuint program = createProgram(objects, 3);
		if(!program)
			return -1;
		return static_cast<int>(program);
	}

	bool ShaderManager::useShader(GLuint program){
		glUseProgram(program);
		CHECK_GL_ERRORS;
		return true;
	}

	GLuint ShaderManager::createProgram(const GLchar *vertexShaderSource, const GLchar *fragmentShaderSource){
		GLuint ProgramObject;   // handles to objects
		GLint  vertCompiled, fragCompiled;    // status values
		GLint linked;

		// Create a vertex shader object and a fragment shader object
		GLuint VertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
		GLuint FragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
		// Load source code strings into shaders
		glShaderSource(VertexShaderObject, 1, &vertexShaderSource, NULL);
		glShaderSource(FragmentShaderObject, 1, &fragmentShaderSource, NULL);
		// Compile the brick vertex shader, and print out
		// the compiler log file.
		glCompileShader(VertexShaderObject);
		CHECK_GL_ERRORS;
		glGetShaderiv(VertexShaderObject, GL_COMPILE_STATUS, &vertCompiled);
		printShaderInfoLog(VertexShaderObject);
		// Compile the brick vertex shader, and print out
		// the compiler log file.
		glCompileShader(FragmentShaderObject);
		CHECK_GL_ERRORS;
		glGetShaderiv(FragmentShaderObject, GL_COMPILE_STATUS, &fragCompiled);
		printShaderInfoLog(FragmentShaderObject);
		if (!vertCompiled || !fragCompiled)
			return 0;
		// Create a program object and attach the two compiled shaders
		ProgramObject = glCreateProgram();
		glAttachShader(ProgramObject, VertexShaderObject);
		glAttachShader(ProgramObject, FragmentShaderObject);
		// Link the program object and print out the info log
		glLinkProgram(ProgramObject);
		CHECK_GL_ERRORS;
		glGetProgramiv(ProgramObject, GL_LINK_STATUS, &linked);
		printProgramInfoLog(ProgramObject);
		if (!linked)
			return 0;
		return ProgramObject;
	}

	GLuint ShaderManager::compile(const char *shaderSource, GLuint shaderType) {
		GLint  compiled;
		GLuint shaderObject = glCreateShader(shaderType);

		glShaderSource(shaderObject, 1, &shaderSource, NULL);

		glCompileShader(shaderObject);
		CHECK_GL_ERRORS;
		glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &compiled);
		printShaderInfoLog(shaderObject);

		return shaderObject;
	}

	GLuint ShaderManager::createProgram(GLuint objects[], int size) {
		GLuint programObject = glCreateProgram();
		for (int i = 0; i < size; ++i)
			if(objects[i])
				glAttachShader(programObject, objects[i]);
		glLinkProgram(programObject);
		CHECK_GL_ERRORS;
		GLint linked;
		glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
		printProgramInfoLog(programObject);
		if (!linked)
			return 0;
		return programObject;
	}

} // aergia namespace