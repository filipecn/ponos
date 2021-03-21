/// Copyright (c) 2021, FilipeCN.
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
///\file memory_dump.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-03-14
///
///\brief

#ifndef PONOS_PONOS_PONOS_LOG_MEMORY_DUMP_H
#define PONOS_PONOS_PONOS_LOG_MEMORY_DUMP_H

#include <ponos/common/defs.h>
#include <ponos/common/str.h>
#include <ponos/numeric/numeric.h>
#include <ponos/common/bitmask_operators.h>
#include <ponos/log/console_colors.h>
#include <iostream>
#include <cstdlib> // system

namespace ponos {

enum class memory_dumper_options {
  none = 0x0,
  // data format
  binary = 0x1,
  decimal = 0x2,
  hexadecimal = 0x4,
  hexii = 0x8,
  // display options
  hide_header = 0x10,
  cache_align = 0x20,
  hide_zeros = 0x40,
  hide_ascii = 0x80,
  save_to_string = 0x100,
  write_to_console = 0x200,
  colored_output = 0x400
};
PONOS_ENABLE_BITMASK_OPERATORS(memory_dumper_options);

/// Auxiliary class for analysing blocks of memory
class MemoryDumper {
public:
  struct Region {
    std::size_t offset{0};
    std::size_t size_in_bytes{0};
    std::size_t count{0};
    std::string color = ConsoleColors::default_color;
    std::vector<Region> sub_regions;
  };
  ///
  /// \tparam T
  /// \param data
  /// \param size
  /// \return
  template<typename T>
  static std::string dumpInfo(const T *data, std::size_t size) {
    auto alignment = alignof(T);
    auto ptr = reinterpret_cast<const u8 * >(data);
    ptrdiff_t down_shift = reinterpret_cast<uintptr_t >(ptr) & (64 - 1);
    uintptr_t aligned_base_address = reinterpret_cast<uintptr_t >(ptr) - down_shift;
    auto size_in_bytes = sizeof(T) * size + down_shift;
    Str s = "Memory Block Information\n";
    s.appendLine("    Address:\t", addressOf(reinterpret_cast<uintptr_t>(data)));
    s.appendLine("    Block Size:\t", sizeof(T) * size, " bytes");
    s.appendLine("  Left Alignment");
    s.appendLine("    Type Alignment:\t", alignment);
    s.appendLine("    Shift:\t", down_shift);
    s.appendLine("    Address:\t", addressOf(reinterpret_cast<uintptr_t>(aligned_base_address)));
    s.appendLine("    Total Block Size:\t", size_in_bytes, " bytes");
    return s.str();
  }
  ///
  /// \tparam T
  /// \param data
  /// \param size
  /// \param bytes_per_row
  /// \param options
  /// \return
  template<typename T>
  static std::string dump(const T *data, std::size_t size, u32 bytes_per_row = 8,
                          memory_dumper_options options = memory_dumper_options::none,
                          const std::vector<Region> &regions = {}) {
    // check options
    auto hide_zeros = testMaskBit(options, memory_dumper_options::hide_zeros);
    auto include_header = !testMaskBit(options, memory_dumper_options::hide_header);
    auto align_data = testMaskBit(options, memory_dumper_options::cache_align);
    auto show_ascii = !testMaskBit(options, memory_dumper_options::hide_ascii);
    auto write_to_console = testMaskBit(options, memory_dumper_options::write_to_console);
    auto save_string = testMaskBit(options, memory_dumper_options::save_to_string);
    auto colored_output = testMaskBit(options, memory_dumper_options::colored_output);
    if (!write_to_console && !save_string)
      write_to_console = true;
    // output string
    Str output_string;
    // address size
    u32 address_digit_count = 8;
    // compute column size for text alignment
    u8 data_digit_count = 2;
    if (testMaskBit(options, memory_dumper_options::decimal))
      data_digit_count = 3;
    else if (testMaskBit(options, memory_dumper_options::binary))
      data_digit_count = 8;
    u8 header_digit_count = countHexDigits(bytes_per_row);
    u8 column_size = std::max(header_digit_count, data_digit_count);
    u8 address_column_size = address_digit_count + 2 + 2; // 0x + \t
    if (include_header) {
      Str s = std::string(address_column_size, ' ');
      for (u32 i = 0; i < bytes_per_row; ++i) {
        auto bs = binaryToHex(i, true, true);
        if (i % 8 == 0)
          s.append(" ");
        s.append(std::setw(column_size), !bs.empty() ? bs : "0", " ");
      }
      if (save_string)
        output_string += s;
      if (write_to_console)
        std::cout << s;
    }
    auto alignment = (align_data) ? 64 : 1;
    auto ptr = reinterpret_cast<const u8 * >(data);
    ptrdiff_t shift = reinterpret_cast<uintptr_t >(ptr) & (alignment - 1);
    uintptr_t aligned_base_address = reinterpret_cast<uintptr_t >(ptr) - shift;
    ptrdiff_t byte_offset = 0;
    auto size_in_bytes = sizeof(T) * size + shift;
    auto line_count = 0;
    while (byte_offset < size_in_bytes) {
      { // ADDRESS
        Str s;
        s.appendLine();
        s.append(addressOf(reinterpret_cast<uintptr_t >((void *) (aligned_base_address + byte_offset))).c_str(),
                 "  ");
        if (save_string)
          output_string += s;
        if (write_to_console) {
          if (colored_output && line_count % 2)
            std::cout << ConsoleColors::dim << s << ConsoleColors::reset_dim;
          else
            std::cout << s;
        }
        line_count++;
      }
      std::string ascii_data;
      for (ptrdiff_t i = 0; i < bytes_per_row; i++, byte_offset++) {
        if (i % 8 == 0) {
          if (write_to_console)
            std::cout << " ";
          if (save_string)
            output_string.append(" ");
        }
        if (aligned_base_address + byte_offset < reinterpret_cast<uintptr_t >(ptr) || byte_offset >= size_in_bytes) {
          if (write_to_console)
            std::cout << std::string(column_size, ' ') + " ";
          if (save_string)
            output_string += std::string(column_size, ' ') + " ";
          ascii_data += '.';
          continue;
        }
        u8 byte = *(reinterpret_cast<u8 *>(aligned_base_address + byte_offset));
        Str s;
        if (!hide_zeros || byte) {
          if (testMaskBit(options, memory_dumper_options::hexadecimal))
            s.append(binaryToHex(byte), " ");
          else if (testMaskBit(options, memory_dumper_options::decimal))
            s.append(std::setfill('0'), std::setw(column_size), static_cast<u32>(byte), ' ');
          else if (testMaskBit(options, memory_dumper_options::binary))
            s.append(byteToBinary(byte), " ");
          else if (testMaskBit(options, memory_dumper_options::hexii))
            s.append(std::string(column_size, ' '), " ");
          else
            s.append(binaryToHex(byte), " ");
        } else
          s.append(std::string(column_size, ' '), " ");

        if (save_string)
          output_string += s;
        if (write_to_console) {
          if (colored_output)
            std::cout << byteColor(byte_offset, regions) << s.str() <<
                      ConsoleColors::default_color << ConsoleColors::reset;
          else
            std::cout << s.str();
        }
        if (std::isalnum(byte))
          ascii_data += byte;
        else
          ascii_data += '.';
      }
      if (show_ascii) {
        if (write_to_console)
          std::cout << "\t|" << ascii_data << "|";
        if (save_string)
          output_string.append("\t|", ascii_data, "|");
      }
    }
    if (save_string)
      output_string += '\n';
    if (write_to_console)
      std::cout << "\n";
    return output_string.str();
  }

private:
  static std::string byteColor(std::size_t byte_index, const std::vector<Region> &regions) {
    std::function<std::string(const std::vector<Region> &, std::size_t, const std::string &)> f;
    f = [&](const std::vector<Region> &subregions, std::size_t byte_offset,
            const std::string &parent_color) -> std::string {
      for (const auto &region : subregions) {
        auto region_start = region.offset;
        auto region_end = region_start + region.size_in_bytes * region.count;
        if (byte_offset >= region_start && byte_offset < region_end) {
          if (region.sub_regions.empty())
            return region.color;
          return f(region.sub_regions, (byte_offset - region_start) % region.size_in_bytes, region.color);
        }
      }
      return parent_color;
    };
    return f(regions, byte_index, ConsoleColors::default_color);
  }

  static std::string addressOf(uintptr_t ptr, u32 digit_count = 8) {
    std::string s;
    // TODO: assuming little endianess
    for (i8 i = 7; i >= 0; --i) {
      auto h = binaryToHex((ptr >> (i * 8)) & 0xff, true);
      s += h.substr(h.size() - 2);
    }
    return "0x" + s.substr(s.size() - digit_count, digit_count);
  }

  template<typename T>
  static std::string binaryToHex(T n, bool uppercase = true, bool strip_leading_zeros = false) {
    static const char digits[] = "0123456789abcdef";
    static const char DIGITS[] = "0123456789ABCDEF";
    std::string s;
    for (int i = sizeof(T) - 1; i >= 0; --i) {
      u8 a = n >> (8 * i + 4) & 0xf;
      u8 b = (n >> (8 * i)) & 0xf;
      if (a)
        strip_leading_zeros = false;
      if (!strip_leading_zeros)
        s += (uppercase) ? DIGITS[a] : digits[a];
      if (b)
        strip_leading_zeros = false;
      if (!strip_leading_zeros)
        s += (uppercase) ? DIGITS[b] : digits[b];
    }
    return s;
  }

  static std::string byteToBinary(byte b) {
    std::string s;
    for (int i = 7; i >= 0; i--)
      s += std::to_string((b >> i) & 1);
    return s;
  }

  template<typename T>
  static u8 countHexDigits(T n) {
    u8 count = 0;
    while (n) {
      count++;
      n >>= 4;
    }
    return count;
  }
};

}

#endif //PONOS_PONOS_PONOS_LOG_MEMORY_DUMP_H
