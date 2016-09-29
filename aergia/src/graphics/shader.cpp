#include "graphics/shader.h"

namespace aergia {

	bool Shader::loadFromFiles(const char *fl...) {
		running = false;
		int program = ShaderManager::instance().loadFromFiles(fl);
		if(program < 0)
			return false;
		programId = static_cast<GLuint>(program);
		return true;
	}

	bool Shader::begin(){
		if(running)
			return true;
		if(!ShaderManager::instance().useShader(programId))
			return false;
		for(auto va : vertexAttributes){
			GLint attribute = glGetAttribLocation(programId, va.name);
			glEnableVertexAttribArray(attribute);
			glVertexAttribPointer(attribute, va.size, GL_FLOAT, GL_FALSE, vDataSize * sizeof(GLfloat), (void*)(va.offset * sizeof(GLfloat)));
		}
		running = true;
		return true;
	}

	void Shader::end(){
		glUseProgram(0);
		running = false;
	}

	void Shader::addVertexAttribute(const char *attribute, int size, int offset){
		VertexAttribute va;
		va.name = attribute;
		va.size = size;
		va.offset = offset;

		vertexAttributes.push_back(va);
	}

	void Shader::setUniform(const char* name, const ponos::mat4 &m){
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniformMatrix4fv(loc, 1, GL_FALSE, &m.m[0][0]);
	}

	void Shader::setUniform(const char* name, const ponos::mat3 &m){
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniformMatrix3fv(loc, 1, GL_FALSE, &m.m[0][0]);
	}

	void Shader::setUniform(const char* name, const ponos::vec4 &v){
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniform4fv(loc, 1, &v.x);
	}

	void Shader::setUniform(const char* name, const ponos::vec3 &v){
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniform3fv(loc, 1, &v.x);
	}

	void Shader::setUniform(const char* name, const ponos::vec2 &v){
		bool wasNotRunning = !running;
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniform2fv(loc, 1, &v.x);
		if(wasNotRunning)
			end();
	}

	void Shader::setUniform(const char* name, int i){
		bool wasNotRunning = !running;
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniform1i(loc, i);
		if(wasNotRunning)
			end();
	}

	void Shader::setUniform(const char* name, float f){
		bool wasNotRunning = !running;
		GLint loc = getUniLoc(name);
		if(loc == -1)
			return;
		glUniform1f(loc, f);
		if(wasNotRunning)
			end();
	}

	GLint Shader::getUniLoc(const GLchar *name){
		if(!ShaderManager::instance().useShader(programId))
			return -1;
		return glGetUniformLocation(programId, name);
	}

} // aergia namespace
