#include <catch2/catch.hpp>

#include <ponos/log/memory_dump.h>
#include <ponos/log/console_colors.h>
#include <ponos/log/debug.h>

using namespace ponos;

TEST_CASE("debug macros", "[log]") {
  SECTION("flow") {
    {
      auto rif = [](int a, int b) -> bool {
        PONOS_RETURN_IF(a == b, true)
        return false;
      };
      auto rnif = [](int a, int b) -> bool {
        PONOS_RETURN_IF_NOT(a == b, true)
        return false;
      };
      REQUIRE(!rif(3, 2));
      REQUIRE(rif(3, 3));
      REQUIRE(rnif(3, 2));
      REQUIRE(!rnif(3, 3));
    }
    {
      auto rif = [](int a, int b) -> bool {
        PONOS_RETURN_IF(a == b, true)
        return false;
      };
      auto rnif = [](int a, int b) -> bool {
        PONOS_RETURN_IF_NOT_WITH_LOG(a == b, true, "message")
        return false;
      };
      REQUIRE(!rif(3, 2));
      REQUIRE(rif(3, 3));
      REQUIRE(rnif(3, 2));
      REQUIRE(!rnif(3, 3));
    }
  }
  PONOS_PING
  PONOS_LOG(3)
  PONOS_LOG_WARNING("warning")
  PONOS_LOG_ERROR("error")
  PONOS_LOG_CRITICAL("critical")
  int a = 0;
  PONOS_LOG_VARIABLE(a)
  PONOS_CHECK_EXP(3 == 3)
  PONOS_CHECK_EXP(3 == 2)
  PONOS_CHECK_EXP_WITH_LOG(3 == 3, "message")
  PONOS_CHECK_EXP_WITH_LOG(3 == 2, "message")
  PONOS_ASSERT(3 == 3)
  PONOS_ASSERT(3 == 2)
  PONOS_ASSERT_WITH_LOG(3 == 3, "message")
  PONOS_ASSERT_WITH_LOG(3 == 2, "message")
}

TEST_CASE("Console Colors", "[log]") {

#define PRINT_COLOR_NAME(COLOR) \
  std::cout << COLOR << #COLOR << std::endl;

  PRINT_COLOR_NAME(ConsoleColors::default_color)
  PRINT_COLOR_NAME(ConsoleColors::black)
  PRINT_COLOR_NAME(ConsoleColors::red)
  PRINT_COLOR_NAME(ConsoleColors::green)
  PRINT_COLOR_NAME(ConsoleColors::yellow)
  PRINT_COLOR_NAME(ConsoleColors::blue)
  PRINT_COLOR_NAME(ConsoleColors::magenta)
  PRINT_COLOR_NAME(ConsoleColors::cyan)
  PRINT_COLOR_NAME(ConsoleColors::light_gray)
  PRINT_COLOR_NAME(ConsoleColors::dark_gray)
  PRINT_COLOR_NAME(ConsoleColors::light_red)
  PRINT_COLOR_NAME(ConsoleColors::light_green)
  PRINT_COLOR_NAME(ConsoleColors::light_yellow)
  PRINT_COLOR_NAME(ConsoleColors::light_blue)
  PRINT_COLOR_NAME(ConsoleColors::light_magenta)
  PRINT_COLOR_NAME(ConsoleColors::light_cyan)
  PRINT_COLOR_NAME(ConsoleColors::white)
  PRINT_COLOR_NAME(ConsoleColors::background_default_color)
  PRINT_COLOR_NAME(ConsoleColors::background_black)
  PRINT_COLOR_NAME(ConsoleColors::background_red)
  PRINT_COLOR_NAME(ConsoleColors::background_green)
  PRINT_COLOR_NAME(ConsoleColors::background_yellow)
  PRINT_COLOR_NAME(ConsoleColors::background_blue)
  PRINT_COLOR_NAME(ConsoleColors::background_magenta)
  PRINT_COLOR_NAME(ConsoleColors::background_cyan)
  PRINT_COLOR_NAME(ConsoleColors::background_light_gray)
  PRINT_COLOR_NAME(ConsoleColors::background_dark_gray)
  PRINT_COLOR_NAME(ConsoleColors::background_light_red)
  PRINT_COLOR_NAME(ConsoleColors::background_light_green)
  PRINT_COLOR_NAME(ConsoleColors::background_light_yellow)
  PRINT_COLOR_NAME(ConsoleColors::background_light_blue)
  PRINT_COLOR_NAME(ConsoleColors::background_light_magenta)
  PRINT_COLOR_NAME(ConsoleColors::background_light_cyan)
  PRINT_COLOR_NAME(ConsoleColors::background_white)

  std::cout << ConsoleColors::background_default_color;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << ConsoleColors::color(r * 8 + i) <<
                std::to_string(r * 8 + i) << " ";
    std::cout << std::endl;
  }

  std::cout << ConsoleColors::black;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << ConsoleColors::background_color(r * 8 + i) <<
                std::to_string(r * 8 + i) << " ";
    std::cout << std::endl;
  }
  std::cout << ConsoleColors::reset << "reset\n";
  std::cout << ConsoleColors::combine(ConsoleColors::blink, ConsoleColors::green) <<
            "blink green " << ConsoleColors::reset << std::endl;
#undef PRINT_COLOR_NAME
}

TEST_CASE("MemoryDumper", "[log]") {
  SECTION("hex") {
    const char s[] = "abcdefghijklmnopqrstuvxzwy";
    MemoryDumper md;
    std::cerr << MemoryDumper::dump(s, sizeof(s));
    const u32 v32[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v32, 5);
  }//
  SECTION("binary") {
    const u8 v8[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v8, 5, 8, memory_dumper_options::binary);
  }//
  SECTION("decimal") {
    const u16 v16[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v16, 5, 8, memory_dumper_options::decimal);
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
    MemoryDumper::dump(data, 4, 8, memory_dumper_options::hide_zeros);
  }//
  SECTION("row size") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    MemoryDumper::dump(v, 8, 24);
  }//
  SECTION("hide header and ascii") {
    u16 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    MemoryDumper::dump(v, 8, 8, memory_dumper_options::hide_header
        | memory_dumper_options::hide_ascii);
  }//
  SECTION("cache align") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cerr << MemoryDumper::dumpInfo(v, 8);
    MemoryDumper::dump(v, 8, 64, memory_dumper_options::cache_align
        | memory_dumper_options::hide_ascii);
  } //
  SECTION("colored output") {
    SECTION("packed members") {
      struct S {
        u64 b;
        u32 a;
        u8 c;
      };
      S v[5] = {{1, 1, 1}, {2, 2, 2},
                {3, 3, 3}, {4, 4, 4}, {5, 5, 5}};
      MemoryDumper::dump(v, 5, 16, memory_dumper_options::colored_output
                             | memory_dumper_options::cache_align,
                         {
                             {0,
                              sizeof(S),
                              5,
                              ConsoleColors::combine(ConsoleColors::yellow, ConsoleColors::dim),
                              {
                                  // a
                                  {
                                      offsetof(S, a),
                                      sizeof(S::a),
                                      1,
                                      ConsoleColors::green,
                                      {}},
                                  // b
                                  {
                                      offsetof(S, b),
                                      sizeof(S::b),
                                      1,
                                      ConsoleColors::red,
                                      {}},
                                  // c
                                  {
                                      offsetof(S, c),
                                      sizeof(S::c),
                                      1,
                                      ConsoleColors::blue,
                                      {}},
                              }},
                         });
    }//
    SECTION("bad alignment") {
      struct S {
        u8 c;
        u64 b;
        u32 a;
      };
      S v[5] = {{1, 1, 1}, {2, 2, 2},
                {3, 3, 3}, {4, 4, 4}, {5, 5, 5}};
      MemoryDumper::dump(v, 5, 16, memory_dumper_options::colored_output
                             | memory_dumper_options::cache_align,
                         {
                             {0,
                              sizeof(S),
                              5,
                              ConsoleColors::combine(ConsoleColors::yellow, ConsoleColors::dim),
                              {
                                  // a
                                  {
                                      offsetof(S, a),
                                      sizeof(S::a),
                                      1,
                                      ConsoleColors::green,
                                      {}},
                                  // b
                                  {
                                      offsetof(S, b),
                                      sizeof(S::b),
                                      1,
                                      ConsoleColors::red,
                                      {}},
                                  // c
                                  {
                                      offsetof(S, c),
                                      sizeof(S::c),
                                      1,
                                      ConsoleColors::blue,
                                      {}},
                              }},
                         });
    }//

  }//
}
