#include "common/file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace ponos {

	int readFile(const char *filename, char **text){
		int count;

		int fd = open(filename, O_RDONLY);
		if (fd == -1)
			return 0;

		size_t size = (size_t) (lseek(fd, 0, SEEK_END) + 1);
		close(fd);
		*text = new char[size];

		FILE *f = fopen(filename, "r");
		if(!f)
			return 0;

		fseek(f, 0, SEEK_SET);
		count = (int) fread(*text, 1, size, f);
		(*text)[count] = '\0';

		if (ferror(f))
			count = 0;

		fclose(f);
		return count;
	}
} // ponos namespace
