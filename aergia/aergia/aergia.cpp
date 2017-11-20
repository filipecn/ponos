#include "aergia.h"

#include <iostream>

namespace aergia {

	bool initialize() {
		static bool initialized = false;
		if(initialized)
			return true;
		int gl_major, gl_minor;
		// Initialize the "OpenGL Extension Wrangler" library
		if(!initGLEW()) {
			std::cout << "glew init failed!\n";
			return false;
		}
		// Make sure that OpenGL 2.0 is supported by the driver
		getGlVersion(&gl_major, &gl_minor);
		printf("GL_VERSION major=%d minor=%d\n", gl_major, gl_minor);
		if (gl_major < 2) {
			printf("GL_VERSION major=%d minor=%d\n", gl_major, gl_minor);
			printf("Support for OpenGL 2.0 is required for this demo...exiting\n");
			//exit(1);
		}
		initialized = true;
		return true;
	}

} // aergia namespace
