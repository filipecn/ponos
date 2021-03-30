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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <iostream>
#include <stack>
#include <algorithm>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#include <direct.h>
#include <fstream>
#include <string>
#include <utility>
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

Path::Path(const Str& path) : path_(path) {}

Path::Path(std::string path) : path_(std::move(path)) {}

Path::Path(const char *const &&path) : path_(path) {}

Path::Path(const Path &other) : path_(other.path_) {}

Path::Path(Path &&other)  noexcept : path_(std::move(other.path_)) {}

Path &Path::operator=(const Path &path) {
  path_ = path.path_;
  return *this;
}

Path &Path::operator+=(const std::string &other) {
  this->join(other);
  return *this;
}

Path &Path::operator+=(const Path &other) {
  this->join(other);
  return *this;
}

Path &Path::cd(const std::string &path) {
  // TODO: what if path represents a file?
  auto current_path = Str::split(path_.str(), separator);
  std::stack<std::string> stack;
  for (const auto &s : current_path)
    stack.push(s);
  auto subpaths = Str::split(path, separator);
  for (const auto &p : subpaths) {
    if (p == ".")
      continue;
    if (p == ".." && !stack.empty())
      stack.pop();
    else if (p != "..")
      stack.push(p);
  }
  path_ = "";
  bool first = true;
  while (!stack.empty()) {
    if (first)
      path_ = stack.top();
    else
      path_ = Str() << stack.top() << separator << path_;
    first = false;
    stack.pop();
  }
  return *this;
}

Path &Path::join(const Path &path) {
  path_ = Str::join({path_.str(), path.path_.str()}, separator);
  path_ = Str::replace_r(path_.str(), separator + separator, separator);
  return *this;
}

bool Path::exists() const {
  return FileSystem::isFile(path_.str()) || FileSystem::isDirectory(path_.str());
}

bool Path::isDirectory() const {
  return FileSystem::isDirectory(path_.str());
}

bool Path::isFile() const {
  return FileSystem::isFile(path_.str());
}

std::vector<std::string> Path::parts() const {
  return Str::split(path_.str(), separator);
}

std::string Path::name() const {
  return FileSystem::basename(path_.str());
}

const std::string &Path::fullName() const {
  return path_.str();
}

std::string Path::extension() const {
  return FileSystem::fileExtension(path_.str());
}

bool Path::match_r(const std::string &regular_expression) const { return false; }

Path Path::cwd() const {
  std::size_t found = path_.str().find_last_of("/\\");
  if (found != std::string::npos)
    return Path(path_.str().substr(0, found));
  return *this;
}

bool Path::mkdir() const {
  return FileSystem::mkdir(path_.str());
}

bool Path::touch() const {
  return FileSystem::touch(path_.str());
}

u64 Path::writeTo(const std::string &content) const { return 0; }

u64 Path::appendTo(const std::string &content) const { return 0; }

std::string Path::read() const {
  if (!isFile())
    return "";
  return FileSystem::readFile(path_.str());
}

Path operator+(const Path &a, const Str &b) {
  Path p = a;
  p.join(b.str());
  return p;
}


bool operator==(const Path &a, const Path &b) {
  return static_cast<std::string>(a) == static_cast<std::string>(b);
}

std::ostream &operator<<(std::ostream &o, const Path &path) {
  o << path.fullName();
  return o;
}

std::string FileSystem::basename(const std::string &path, const std::string &suffix) {
  std::size_t found = path.find_last_of("/\\");
  std::string base_name = (found != std::string::npos) ? path.substr(found + 1) : path;
  if (!suffix.empty() && suffix.size() <= base_name.size()
      && base_name.substr(base_name.size() - suffix.size()) == suffix)
    return base_name.substr(0, base_name.size() - suffix.size());
  return base_name;
}

std::vector<std::string> FileSystem::basename(const std::vector<std::string> &paths, const std::string &suffix) {
  std::vector<std::string> base_names;
  for (const auto &p : paths)
    base_names.emplace_back(basename(p, suffix));
  return base_names;
}

std::string FileSystem::fileExtension(const std::string &filename) {
  std::string extension = filename;
  size_t i = extension.rfind('.', extension.length());
  if (i != std::string::npos)
    extension = extension.substr(i + 1, extension.length());
  else
    extension = "";
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

std::string FileSystem::readFile(const Path &path) {
  std::string content;
  std::ifstream file(path.fullName(), std::ios::in);
  if (file.good()) {
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
    file.close();
  }
  return content;
}

std::vector<std::string> FileSystem::readLines(const Path &path) {
  std::vector<std::string> lines;
  std::ifstream file(path.fullName(), std::ios::in);
  std::string line;
  while (std::getline(file, line))
    lines.emplace_back(std::move(line));
  return lines;
}

bool FileSystem::fileExists(const Path &path) {
  if (FILE *file = fopen(path.fullName().c_str(), "r")) {
    fclose(file);
    return true;
  }
  return false;
}

bool FileSystem::touch(const Path &path_to_file) {
  std::ofstream file(path_to_file.fullName());
  if (file.good()) {
    file.close();
    return true;
  }
  return false;
}

u64 FileSystem::writeFile(const Path &path, const std::vector<char> &content, bool is_binary) {
  std::ios_base::openmode flags = std::ofstream::out;
  if (is_binary)
    flags |= std::ofstream::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::writeFile(const Path &path, const std::string &content, bool is_binary) {
  std::ios_base::openmode flags = std::ios::out;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::writeLine(const Path &path, const std::string &line, bool is_binary) {
  std::ios_base::openmode flags = std::ios::out;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file << line << std::endl;
    file.close();
    return line.size();
  }
  return 0;
}

u64 FileSystem::appendToFile(const Path &path, const std::vector<char> &content, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file.write(content.data(), content.size());
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::appendToFile(const Path &path, const std::string &content, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file << content;
    file.close();
    return content.size();
  }
  return 0;
}

u64 FileSystem::appendLine(const Path &path, const std::string &line, bool is_binary) {
  auto flags = std::ios::out | std::ios::app;
  if (is_binary)
    flags |= std::ios::binary;
  std::ofstream file(path.fullName(), flags);
  if (file.good()) {
    file << line << std::endl;
    file.close();
    return line.size();
  }
  return 0;
}

bool FileSystem::isFile(const Path &path) {
  struct stat st{};
  if (!stat(path.fullName().c_str(), &st))
    return ((st.st_mode) & S_IFMT) == S_IFREG;
  return false;
}

bool FileSystem::isDirectory(const Path &dir_name) {
  struct stat st{};
  if (!stat(dir_name.fullName().c_str(), &st))
    return ((st.st_mode) & S_IFMT) == S_IFDIR;
  return false;
}

std::vector<Path> FileSystem::ls(const Path &path, ls_options options) {
  std::vector<Path> l;
  std::function<void(const std::string &)> ls_;
#ifdef WIN32

  ls_ = [&](const std::string &directory_path) {
    for (const auto &entry : std::filesystem::directory_iterator(directory_path)) {
      auto full_path = entry.path().string();
      bool is_directory = entry.is_directory();
      if ((options & ls_options::recursive) == ls_options::recursive && is_directory)
        ls_(full_path);
      if ((options & ls_options::directories) == ls_options::directories && !is_directory)
        continue;
      if ((options & ls_options::files) == ls_options::files && is_directory)
        continue;
      l.emplace_back(normalizePath(full_path));
    }
  };
#else
  ls_ = [&](const std::string &directory_path) {
    DIR *dir = opendir(directory_path.c_str());
    if (dir != nullptr) {
      struct dirent *dp;
      while ((dp = readdir(dir)) != nullptr) {
        if (std::string(dp->d_name) == "." || std::string(dp->d_name) == "..")
          continue;
        auto full_path = directory_path + (Str() << "/" << dp->d_name);
        bool is_directory = isDirectory(full_path);
        if ((options & ls_options::recursive) == ls_options::recursive && is_directory)
          ls_(full_path.fullName());
        if ((options & ls_options::directories) == ls_options::directories && !is_directory)
          continue;
        if ((options & ls_options::files) == ls_options::files && is_directory)
          continue;
        l.emplace_back(full_path);
      }
      closedir(dir);
    }
  };
#endif
  ls_(path.fullName());
  if ((options & (ls_options::sort | ls_options::reverse_sort | ls_options::group_directories_first))
      != ls_options::none) {
    bool reverse_order = (options & ls_options::reverse_sort) == ls_options::reverse_sort;
    bool group_directories = (options & ls_options::group_directories_first) == ls_options::group_directories_first;
    auto cmp = [&](const Path &a, const Path &b) -> bool {
      if (group_directories) {
        bool a_is_directory = a.isDirectory();
        bool b_is_directory = b.isDirectory();
        if (a_is_directory && b_is_directory)
          return reverse_order ? a.fullName() > b.fullName() : a.fullName() < b.fullName();
        return a_is_directory;
      }
      return reverse_order ? a.fullName() > b.fullName() : a.fullName() < b.fullName();
    };
    std::sort(l.begin(), l.end(), cmp);
  }
  return l;
}

bool FileSystem::mkdir(const Path &path) {
  auto mk_single_dir = [](const std::string &path_) -> int {
    std::string npath = normalizePath(path_);

    struct stat st;
    int status = 0;

    if (stat(path_.c_str(), &st) != 0) {
#ifdef WIN32
      status = _mkdir(path_.c_str());
#else
      status = ::mkdir(path_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
      if (status != 0 && errno != EEXIST)
        status = -1;
    } else if (!(S_IFDIR & st.st_mode)) {
      errno = ENOTDIR;
      status = -1;
    }

    return status;
  };
  char *pp;
  char *sp;
  int status;
#ifdef WIN32
  char *copyOfPath = _strdup(path.c_str());
#else
  char *copyOfPath = strdup(path.fullName().c_str());
#endif

  status = 0;
  pp = copyOfPath;
  pp = pp + 3;        // Cut away Drive:
  while ((status == 0) && (((sp = strchr(pp, '/')) != 0) || ((sp = strchr(pp, '\\')) != 0))) {
    if (sp != pp) {
      *sp = '\0';
      status = mk_single_dir(copyOfPath);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if (status == 0)
    status = mk_single_dir(path.fullName());
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
  while (index < tokens.size()) {
    if ((tokens[index] == "..") && (index > 0)) {
      tokens.erase(tokens.begin() + index - 1, tokens.begin() + index + 1);
      index -= 2;
    }
    index++;
  }
  result = "";
  if (path[0] == '/')
    result = "/";
  result = result + tokens[0];
  for (unsigned int i = 1; i < tokens.size(); i++)
    result += "/" + tokens[i];
  return result;
}

bool FileSystem::copyFile(const Path &source, const Path &destination) {
  const size_t bufferSize = 8192;
  char buffer[bufferSize];
  size_t size;

  FILE *sourceFile = fopen(source.fullName().c_str(), "rb");
  FILE *destFile = fopen(destination.fullName().c_str(), "wb");

  if ((sourceFile == nullptr) || (destFile == nullptr))
    return false;

  while ((size = fread(buffer, 1, bufferSize, sourceFile))) {
    fwrite(buffer, 1, size, destFile);
  }

  fclose(sourceFile);
  fclose(destFile);

  return true;
}

std::vector<Path> FileSystem::find(const Path &path, const std::string &pattern, find_options options) {
  std::vector<Path> found;
  ls_options lso = ls_options::files;
  if ((options & find_options::recursive) == find_options::recursive)
    lso = lso | ls_options::recursive;
  if ((options & find_options::sort) == find_options::sort)
    lso = lso | ls_options::sort;
  const auto &l = ls(path, lso);
  for (const auto &p : l)
    if (Str::contains_r(p.fullName(), pattern))
      found.emplace_back(p);
  return found;
}

}
