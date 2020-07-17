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
///\file arg_parser.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-13-07
///
///\brief Simple argument parser

#include <ponos/common/arg_parser.h>

#include <iostream>
#include <iomanip>
#include <utility>

namespace ponos {

ArgParser::ArgParser(std::string bin, std::string description)
    : bin_{std::move(bin)}, description_{std::move(description)} {}

bool ArgParser::parse(int argc, char **argv) {
  // build map of argument names
  for(u64 i = 0; i < arguments_.size(); ++i)
    for(const auto& n : arguments_[i].names)
      m_[n] = i;
  int current_argument = -1;
  // nameless arguments follow the arguments order
  u64 current_unknown = 0;
  for(int i = 1; i < argc; ++i) {
    const auto& it = m_.find(std::string(argv[i]));
    if(it != m_.end()) {
      current_argument = it->second;
      arguments_[current_argument].given = true;
    } else if(current_argument >= 0) {
      arguments_[current_argument].values.emplace_back(argv[i]);
      current_argument = -1;
    } else if(current_unknown < arguments_.size()) {
      while(current_unknown < arguments_.size() && arguments_[current_unknown].given)
        current_unknown++;
      arguments_[current_unknown].values.emplace_back(argv[i]);
      arguments_[current_unknown].given = true;
    }
  }
  // check if all required arguments were given
  for(const auto& a:  arguments_)
    if(a.required && !a.given)
      return false;
  return true;
}

void ArgParser::addArgument(const std::string &name, const std::string &description, bool is_required) {
  Argument arg{};
  arg.description = description;
  arg.required = is_required;
  arg.names = {name};
  arguments_.emplace_back(arg);
}

void ArgParser::printHelp() const {
  std::cout << "Usage: " << bin_ << std::endl;
  std::cout << "\t" << description_ << std::endl;
  for(const auto& a : arguments_) {
    std::string argument_name = a.names[0];
    for(u64 i = 1; i < a.names.size(); ++i)
      argument_name += ", " + a.names[i];
    std::cout << "    " << std::setw(23) << std::left << argument_name << std::setw(23)
              << a.description;
    if(a.required)
      std::cout << "(required)";
    std::cout << std::endl;
  }
}

bool ArgParser::check(const std::string &name) const {
  const auto& it = m_.find(name);
  return it != m_.end() && arguments_[it->second].given;
}

}