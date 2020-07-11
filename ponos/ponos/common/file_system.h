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
#include <vector>
#include <string>

namespace ponos {

/// Set of usefull functions to manipulate files and directories
class FileSystem {
public:
  /// \param filename path/to/filename.extension
  /// \return file's extension, if any (after last '.')
  static std::string fileExtension(const std::string& filename);
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
  static std::string readFile(const std::string& filename);
  /// Writes content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::string& filename,
      const std::vector<char>& content, bool is_binary = false);
  /// Writes content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 writeFile(const std::string& filename,
                       const std::string& content, bool is_binary = false);
  /// Appends content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::string& filename,
      const std::vector<char>& content, bool is_binary = false);
  /// Appends content to file.
  /// \param filename **[in]** path/to/file.ext
  /// \param content
  /// \param is_binary **[in | default = false]** write in binary mode
  /// \return number of bytes successfully written.
  static u64 appendToFile(const std::string& filename,
                          const std::string& content, bool is_binary = false);
  /// Checks if file exists
  /// \param filename /path/to/file.ext
  /// \return **true** if file exists
  static bool fileExists(const std::string& filename);
  /// Checks if filename corresponds to a file.
  /// \param filename /path/to/file.ext
  /// \return **true** if filename points to a file.
  static bool isFile(const std::string& filename);
  /// Checks if dir_name corresponds to a directory.
  /// \param dir_name **[in]** /path/to/directory
  /// \return **true** if dir_name points to a directory.
  static bool isDirectory(const std::string& dir_name);
  /// Lists files inside a directory
  /// \param path **[in]** path/to/directory
  /// \return list of filenames
  static std::vector<std::string> ls(const std::string& path);
  /// Recursively creates the path of directories
  /// \param path path/to/directory
  /// \return true on success success
  static bool mkdir(const std::string& path);
  ///
  /// \param source
  /// \param destination
  /// \return
  static bool copyFile(const std::string& source, const std::string& destination);
  ///
  /// \param path
  /// \param with_backslash
  /// \return
  static std::string normalizePath(const std::string& path, bool with_backslash = false);
};

}

#endif //PONOS_FILE_SYSTEM_H
