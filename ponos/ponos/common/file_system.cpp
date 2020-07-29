/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file file_system.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief

#include "file_system.h"
#include <ponos/common/str.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>

#ifdef _WIN32
#include <fstream>
#include <string>
#include <windows.h>
// Allow use of freopen() without compilation warnings/errors.
// For use with custom allocated console window.
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#else
#include <unistd.h>
#include <dirent.h>
#endif

namespace ponos {

std::string FileSystem::fileExtension(const std::string &filename) {
  std::string extension = filename;
  size_t i = extension.rfind('.', extension.length());
  if (i != std::string::npos)
    extension = extension.substr(i + 1, extension.length());
  return extension;
}

#ifdef WIN32
// TODO handle errors
int readFile(const char *filename, char **text) {
  std::ifstream file(filename);
  std::string str;
  std::string contents;
  while (std::getline(file, str)) {
    contents += str;
    contents.push_back('\n');
  }
  if (!contents.size())
    return 0;
  *text = new char[contents.size() + 1];
  std::strcpy(*text, contents.c_str());
  (*text)[contents.size()] = '\0';
  return contents.size();
}
#endif

#ifndef WIN32

u64 FileSystem::readFile(const char *filename, char **text) {
  u64 count_;

  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return 0;

  u64 size = (u64) (lseek(fd, 0, SEEK_END) + 1);
  close(fd);
  *text = new char[size];

  FILE *f = fopen(filename, "r");
  if (!f)
    return 0;

  fseek(f, 0, SEEK_SET);
  count_ = (int) fread(*text, 1, size, f);
  (*text)[count_] = '\0';

  if (ferror(f))
    count_ = 0;

  fclose(f);
  return count_;
}

#endif

std::vector<unsigned char> FileSystem::readBinaryFile(const char *filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return std::vector<unsigned char>();
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto bytes = std::vector<unsigned char>(size);
  file.read(reinterpret_cast<char *>(&bytes[0]), size);
  file.close();
  return bytes;
}

std::string FileSystem::readFile(const std::string &filename) {
  std::string content;
  std::ifstream file(filename, std::ios::in);
  if(file.good()) {
    file >> content;
    file.close();
  }
  return content;
}

bool FileSystem::fileExists(const std::string &filename) {
  if (FILE *file = fopen(filename.c_str(), "r")) {
    fclose(file);
    return true;
  }
    return false;
}

u64 FileSystem::writeFile(const std::string &filename, const std::vector<char> &content, bool is_binary) {
  auto flags = std::ios::out;
  if(is_binary)
    flags |= std::ios::binary;
  std::ofstream file(filename, flags);
  if(file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::writeFile(const std::string &filename, const std::string &content, bool is_binary) {
  auto flags = std::ios::out;
  if(is_binary)
    flags |= std::ios::binary;
  std::ofstream file(filename, flags);
  if(file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::appendToFile(const std::string &filename, const std::vector<char> &content, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if(is_binary)
    flags |= std::ios::binary;
  std::ofstream file(filename, flags);
  if(file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::appendToFile(const std::string &filename, const std::string &content, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(filename, flags);
  if (file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

bool FileSystem::isFile(const std::string &filename) {
  struct stat st{};
  if (!stat(filename.c_str(), &st))
    return ((st.st_mode) & S_IFMT) == S_IFREG;
  return false;
}

bool FileSystem::isDirectory(const std::string &dir_name) {
  struct stat st{};
  if (!stat(dir_name.c_str(), &st))
    return ((st.st_mode) & S_IFMT) == S_IFDIR;
  return false;
}

std::vector<std::string> FileSystem::ls(const std::string &path) {
  std::vector<std::string> l;
#ifdef WIN32
  std::string p = path + "\\*";
			WIN32_FIND_DATA data;
			HANDLE hFind = FindFirstFile(p.c_str(), &data);
			if (hFind  != INVALID_HANDLE_VALUE)
			{
				do
				{
					res.push_back(data.cFileName);
				}
				while (FindNextFile(hFind, &data) != 0);
				FindClose(hFind);
				return true;
			}
			return false;
#else
  DIR* dir = opendir(path.c_str());
  if (dir != nullptr)
  {
    struct dirent *dp;
    while ((dp = readdir(dir)) != nullptr)
      l.emplace_back(dp->d_name);
    closedir(dir);
    return l;
  }
  return l;
#endif
}

bool FileSystem::mkdir(const std::string &path) {
  auto mk_single_dir = [](const std::string& path_) -> int {
    std::string npath = normalizePath(path_);

    struct stat st;
    int status = 0;

    if (stat(path_.c_str(), &st) != 0)
    {
#if WIN32
      status = _mkdir(path.c_str());
#else
      status = ::mkdir(path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
      if (status != 0 && errno != EEXIST)
        status = -1;
    }
    else if (!(S_IFDIR & st.st_mode))
    {
      errno = ENOTDIR;
      status = -1;
    }

    return status;
  };
  char *pp;
  char *sp;
  int  status;
#ifdef WIN32
  char *copyOfPath = _strdup(path.c_str());
#else
  char *copyOfPath = strdup(path.c_str());
#endif

  status = 0;
  pp = copyOfPath;
  pp = pp + 3;		// Cut away Drive:
  while ((status == 0) && (((sp = strchr(pp, '/')) != 0) || ((sp = strchr(pp, '\\')) != 0)))
  {
    if (sp != pp)
    {
      *sp = '\0';
      status = mk_single_dir(copyOfPath);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if (status == 0)
    status = mk_single_dir(path);
  free(copyOfPath);
  // TODO handle error
  return status == 0;
}

std::string FileSystem::normalizePath(const std::string &path, bool with_backslash) {
  if (path.size() == 0)
    return path;
  std::string result = path;
  std::replace(result.begin(), result.end(), '\\', '/');
  std::vector<std::string> tokens = Str::split(result, "/");
  unsigned int index = 0;
  while (index < tokens.size())
  {
    if ((tokens[index] == "..") && (index > 0))
    {
      tokens.erase(tokens.begin() + index - 1, tokens.begin() + index + 1);
      index-=2;
    }
    index++;
  }
  result = "";
  if (path[0] == '/')
    result = "/";
  result = result + tokens[0];
  for (unsigned int i = 1; i < tokens.size(); i++)
    result = result + "/" + tokens[i];
  return result;
}

bool FileSystem::copyFile(const std::string &source, const std::string &destination) {
  const size_t bufferSize = 8192;
  char buffer[bufferSize];
  size_t size;

  FILE* sourceFile = fopen(source.c_str(), "rb");
  FILE* destFile = fopen(destination.c_str(), "wb");

  if ((sourceFile == NULL) || (destFile == NULL))
    return false;

  while (size = fread(buffer, 1, bufferSize, sourceFile))
  {
    fwrite(buffer, 1, size, destFile);
  }

  fclose(sourceFile);
  fclose(destFile);

  return true;
}

}
