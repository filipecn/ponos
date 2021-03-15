#include <catch2/catch.hpp>

#include <ponos/log/memory_dump.h>

using namespace ponos;

TEST_CASE("MemoryDumper", "[log]") {
  SECTION("hex") {
    const char s[] = "abcdefghijklmnopqrstuvxzwy";
    MemoryDumper md;
    std::cerr << MemoryDumper::dump(s, sizeof(s));
    const u32 v32[] = {1, 2, 3, 4, 5};
    std::cerr << MemoryDumper::dump(v32, 5);
  }//
  SECTION("binary") {
    const u8 v8[] = {1, 2, 3, 4, 5};
    std::cerr << MemoryDumper::dump(v8, 5, 8, memory_dumper_options::binary);
  }//
  SECTION("decimal") {
    const u16 v16[] = {1, 2, 3, 4, 5};
    std::cerr << MemoryDumper::dump(v16, 5, 8, memory_dumper_options::decimal);
  }//
  SECTION("struct") {
    struct Data {
      u8 a;
      u32 b;
      u16 c;
    };
    const Data data[] = {{1, 1, 1},
                         {2, 2, 2},
                         {3, 3, 3},
                         {4, 4, 4}};
    std::cerr << MemoryDumper::dump(data, 4);
  }//
  SECTION("hide zeros") {
    struct Data {
      u32 b;
      u8 a;
      u16 c;
    };
    const Data data[] = {{1, 1, 1},
                         {2, 2, 2},
                         {3, 3, 3},
                         {4, 4, 4}};
    std::cerr << MemoryDumper::dump(data, 4, 8, memory_dumper_options::hide_zeros);
  }//
  SECTION("row size") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cerr << MemoryDumper::dump(v, 8, 24);
  }//
  SECTION("hide header and ascii") {
    u16 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cerr << MemoryDumper::dump(v, 8, 8, memory_dumper_options::hide_header
        | memory_dumper_options::hide_ascii);
  }//
  SECTION("cache align") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cerr << MemoryDumper::dumpInfo(v, 8);
    std::cerr << MemoryDumper::dump(v, 8, 64, memory_dumper_options::cache_align
        | memory_dumper_options::hide_ascii);
  } //
}