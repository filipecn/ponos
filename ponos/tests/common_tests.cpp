#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Str", "[common]") {
  SECTION("split") {
    std::string a = "1 22 3 44 5";
    auto s = Str::split(a);
    REQUIRE(s.size() == 5);
    REQUIRE(s[0] == "1");
    REQUIRE(s[1] == "22");
    REQUIRE(s[2] == "3");
    REQUIRE(s[3] == "44");
    REQUIRE(s[4] == "5");
  }
  SECTION("split with delimiter") {
    std::string a = "1 2, 3,4, 5";
    auto s = Str::split(a, ",");
    REQUIRE(s.size() == 4);
    REQUIRE(s[0] == "1 2");
    REQUIRE(s[1] == " 3");
    REQUIRE(s[2] == "4");
    REQUIRE(s[3] == " 5");
  }
  SECTION("concat") {
    std::string a = Str::concat("a", " ", 2, "b");
    REQUIRE(a == "a 2b");
  }
}

TEST_CASE("FileSystem", "[common]") {
  SECTION("file extension") {
    REQUIRE(FileSystem::fileExtension("path/to/file.ext4") == "ext4");
  }
 SECTION("read invalid file") {
   REQUIRE(!FileSystem::fileExists("invalid__file"));
   REQUIRE(FileSystem::readFile("invalid___file").empty());
    REQUIRE(FileSystem::readBinaryFile("invalid___file").empty());
 }
 SECTION("isFile and isDirectory") {
   REQUIRE(FileSystem::writeFile("filesystem_test_file.txt","test") == 4);
   REQUIRE(FileSystem::isFile("filesystem_test_file.txt"));
   REQUIRE(FileSystem::mkdir("path/to/dir"));
   REQUIRE(FileSystem::isDirectory("path/to/dir"));
 }
 SECTION("copy file") {
    REQUIRE(FileSystem::writeFile("source", "source_content") > 0);
    REQUIRE(FileSystem::copyFile("source", "destination"));
    REQUIRE(FileSystem::fileExists("destination"));
    REQUIRE(FileSystem::readFile("destination") == "source_content");
  }
  SECTION("append") {
    REQUIRE(FileSystem::writeFile("append_test", "") == 0);
    REQUIRE(FileSystem::fileExists("append_test"));
    REQUIRE(FileSystem::readFile("append_test").empty());
    REQUIRE(FileSystem::appendToFile("append_test", "append_content"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content");
    REQUIRE(FileSystem::appendToFile("append_test", "123"));
    REQUIRE(FileSystem::readFile("append_test") == "append_content123");
  }
}

TEST_CASE("Index", "[common][index]") {
  { // Index2
    index2 a;
    index2 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  { // Index2Range
    int cur = 0;
    for (auto index : Index2Range<i32>(10, 10)) {
      REQUIRE(cur % 10 == index.i);
      REQUIRE(cur / 10 == index.j);
      cur++;
    }
    REQUIRE(cur == 10 * 10);
  }
  { // Index3
    index3 a;
    index3 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  { // Index3Range
    int cur = 0;
    for (auto index : Index3Range<i32>(10, 10, 10)) {
      REQUIRE((cur % 100) % 10 == index.i);
      REQUIRE((cur % 100) / 10 == index.j);
      REQUIRE(cur / 100 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
    Index3Range<i32> range(index3(-5, -5, -5), index3(5, 5, 5));
    REQUIRE(range.size().total() == 10 * 10 * 10);
    cur = 0;
    for (auto index : range) {
      REQUIRE((cur % 100) % 10 - 5 == index.i);
      REQUIRE((cur % 100) / 10 - 5 == index.j);
      REQUIRE(cur / 100 - 5 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
  }
}