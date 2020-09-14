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
///\file file_system.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief

#ifndef PONOS_FILE_SYSTEM_H
#define PONOS_FILE_SYSTEM_H

#include <ponos/common/defs.h>
#include <ponos/common/bitmask_operators.h>
#include <vector>
#include <string>
#include <ostream>

namespace ponos {

/// Representation of a directory/file in the filesystem
class Path {
public:
  Path() = default;
  /// \param path
  explicit Path(std::string path);
  explicit operator std::string() const { return path_; }
  Path &operator=(const std::string &path);
  Path &operator+=(const std::string &other);
  Path &operator+=(const Path &other);
  Path &join(const Path &path);
  Path &join(const std::string &path);
  /// Jumpt to path
  /// \param path
  Path &cd(const std::string &path);
  [[nodiscard]] bool exists() const;
  bool isDirectory() const;
  bool isFile() const;
  [[nodiscard]] std::vector<std::string> parts() const;
  [[nodiscard]] std::string name() const;
  [[nodiscard]] const std::string &fullName() const;
  [[nodiscard]] std::string extension() const;
  bool match_r(const std::string &regular_expression) const;
  Path cwd() const;
  bool mkdir() const;
  bool touch() const;
  u64 writeTo(const std::string &content) const;
  u64 appendTo(const std::string &content) const;
  std::string read() const;

  std::string separator{"/"};
private:
  std::string path_;
};

Path operator+(const Path &a, const Path &b);
Path operator+(const std::string &a, const Path &b);
Path operator+(const Path &a, const std::string &b);
bool operator==(const Path &a, const Path &b);
bool operator==(const std::string &a, const Path &b);
bool operator==(const Path &a, const std::string &b);
bool operator==(const char *a, const Path &b);
bool operator==(const Path &a, const char *b);
std::ostream &operator<<(std::ostream &o, const Path &path);

/// List of options for ls method
enum class ls_options {
  none = 0x0,
  sort = 0x1,
  reverse_sort = 0x2,
  directories = 0x4,
  files = 0x8,
  group_directories_first = 0x10,
  recursive = 0x20,
};
ENABLE_BITMASK_OPERATORS(ls_options);
enum class find_options {
  none = 0x0,
  recursive = 0x1,
  sort = 0x2,
};
ENABLE_BITMASK_OPERATORS(find_options);

/// Set of useful functions to manipulate files and directories
class FileSystem {
public:
  /// Strips directory and suffix from filenames
  /// \param paths **[in]** {/path/to/filename1suffix,...}
  /// \param suffix **[in | optional]**
  /// \return {filename1, filename2, ...}
  static std::vector<std::string> basename(const std::vector<std::string> &paths, const std::string &suffix = "");
  /// Strips directory and suffix from filename
  /// \param path **[in]** /path/to/filenamesuffix
  /// \param suffix **[in | optional]**
  /// \return filename
  static std::string basename(const std::string &path, const std::string &suffix = "");
  /// \param filename path/to/filename.extension
  /// \return file's extension, if any (after last '.')
  static std::string fileExtension(const std::string &filename);
  /// \brief loads contents from file
  /// \param filename **[in]** path/to/file.
  /// \param text     **[out]** receives file content.
  /// \return number of bytes successfully read.
  static u64 readFile(const char *filename, char **text);
  /// \param filename **[in]** path/to/file.ext
  /// \return vector of bytes read
  static std::vector<unsigned char> readBinaryFile(const char *filename);
  /// \param filename **[in]** path/to/file.ext
  /// \return file's content
  static std::string readFile(const std::string &filename);
  /// \param **[in]** filename path/to/file.ext
  /// \return file's content
  static std::string readFile(const Path &filename);
  /// Creates an empty file or access it.
  /// \param filename valid path/to/file.ext
  /// \return **true** if success
  static bool touch(const std::string &filename);
  /// Creates an empty file or access it.
  /// \param path_to_file valid file path
  /// \return **true** if success
  static bool touch(const Path &path_to_file);
  /// Writes content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::string &filename,
                       const std::vector<char> &content, bool is_binary = false);
  /// Writes content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::string &filename,
                       const std::string &content, bool is_binary = false);
  /// Appends content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::string &filename,
                          const std::vector<char> &content, bool is_binary = false);
  /// Appends content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::string &filename,
                          const std::string &content, bool is_binary = false);
  /// Checks if file exists
  /// \param filename /path/to/file.ext
  /// \return **true** if file exists
  static bool fileExists(const std::string &filename);
  /// Checks if filename corresponds to a file.
  /// \param filename /path/to/file.ext
  /// \return **true** if filename points to a file.
  static bool isFile(const std::string &filename);
  /// Checks if dir_name corresponds to a directory.
  /// \param dir_name **[in]** /path/to/directory
  /// \return **true** if dir_name points to a directory.
  static bool isDirectory(const std::string &dir_name);
  /// Lists files inside a directory
  /// \param path **[in]** path/to/directory
  /// \param options **[in | ls_options::none]** options based on ls command:
  ///     none = the default behaviour;
  ///     sort = sort paths following lexicographical order;
  ///     reverse_sort = sort in reverse order;
  ///     directories = list only directories;
  ///     files = list only files;
  ///     group_directories_first = directories come first in sorting;
  ///     recursive = list directories contents;
  /// \return list of paths
  static std::vector<Path> ls(const std::string &path, ls_options options = ls_options::none);
  /// Recursively creates the path of directories
  /// \param path path/to/directory
  /// \return true on success success
  static bool mkdir(const std::string &path);
  ///
  /// \param source
  /// \param destination
  /// \return
  static bool copyFile(const std::string &source, const std::string &destination);
  ///
  /// \param path
  /// \param with_backslash
  /// \return
  static std::string normalizePath(const std::string &path, bool with_backslash = false);
  /// Search for files in a directory hierarchy
  /// \param path root directory
  /// \param pattern **[in | ""]** regular expression
  /// \param options **[in | find_options::none]**
  ///     none = default behaviour;
  ///     recursive = recursively search on directories bellow **path**
  /// \return
  static std::vector<Path> find(const std::string &path,
                                const std::string &pattern,
                                find_options options = find_options::none);
};

}

#endif //PONOS_FILE_SYSTEM_H
