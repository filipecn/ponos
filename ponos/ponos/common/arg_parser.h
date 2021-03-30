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
///\file arg_parser.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-13-07
///
///\brief Simple argument parser
/// inspired in https://github.com/jamolnng/argparse/blob/develop/argparse.h

#ifndef PONOS_PONOS_PONOS_COMMON_ARG_PARSER_H
#define PONOS_PONOS_PONOS_COMMON_ARG_PARSER_H

#include <ponos/common/defs.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>

namespace ponos {

/// Command line argument parser
class ArgParser {
public:
  /// \param bin executable name
  /// \param description program description
  explicit ArgParser(std::string bin = "", std::string description = "");
  ///
  /// \param argc
  /// \param argv
  bool parse(int argc, const char**argv, bool verbose_parsing = false);
  ///
  /// \param name
  /// \param description
  /// \param required
  void addArgument(const std::string& name,
                   const std::string & description = "",
                   bool is_required = false);
  ///
  /// \tparam T
  /// \param name
  /// \param default_value
  /// \return
  template<typename T>
  T get(const std::string& name, const T& default_value = {}) const {
      const auto& it = m_.find(name);
      if(it == m_.end() || !arguments_[it->second].given || arguments_[it->second].values.empty())
        return default_value;
    std::istringstream in(arguments_[it->second].values[0]);
    T t{};
    in >> t;
    return t;
  }
    /// Checks if an argument was given
    /// \param name argument's name
    /// \return true if argument was given
  [[nodiscard]] bool check(const std::string& name) const;

  void printHelp() const;

private:
  struct Argument {
    std::string description{};
    std::vector<std::string> names{};
    std::vector<std::string> values{};
    bool required{false};
    bool given{false};
  };
  std::map<std::string, u64> m_;
  std::vector<Argument> arguments_;
  std::string bin_;
  std::string description_;
};

}

#endif //PONOS_PONOS_PONOS_COMMON_ARG_PARSER_H
