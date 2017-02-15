#include "common/file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>

#ifdef _WIN32
#include <windows.h>
// Allow use of freopen() without compilation warnings/errors.
// For use with custom allocated console window.
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#else
#include <unistd.h>
#endif

namespace ponos {

  #ifndef WIN32
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
  #endif
} // ponos namespace
