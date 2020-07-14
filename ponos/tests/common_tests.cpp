#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("ArgParser") {
  SECTION("simple") {
    char * argv[3] = {"bin", "int_argument", "3"};
    ArgParser parser("test_bin", "test bin description");
    parser.addArgument("int_argument", "int argument description");
    parser.addArgument("arg2");
    REQUIRE(parser.parse(3, argv));
    REQUIRE(parser.get<int>("int_argument", 0) == 3);
    REQUIRE(!parser.check("arg2"));
  }
  SECTION("required") {
    char * argv[3] = {"bin", "int_argument", "3"};
    ArgParser parser;
    parser.addArgument("req","", true);
    parser.addArgument("int_argument");
    REQUIRE(!parser.parse(3, argv));

  }
  SECTION("positional arguments") {
    char*argv[5] = {"bin", "4", "arg", "1", "2"};
    ArgParser parser;
    parser.addArgument("a0");
    parser.addArgument("a1");
    parser.addArgument("a2");
    parser.addArgument("a3");
    REQUIRE(parser.parse(5, argv));
    REQUIRE(parser.get<int>("a0") == 4);
    REQUIRE(parser.get<std::string>("a1") == "arg");
    REQUIRE(parser.get<int>("a2") == 1);
    REQUIRE(parser.get<int>("a3") == 2);
    REQUIRE(parser.check("a0"));
    REQUIRE(parser.check("a1"));
    REQUIRE(parser.check("a2"));
    REQUIRE(parser.check("a3"));
  }
  SECTION("positional arguments mixed") {
    char*argv[5] = {"bin", "4", "a1", "1", "2"};
    ArgParser parser;
    parser.addArgument("a0");
    parser.addArgument("a1");
    parser.addArgument("a2");
    REQUIRE(parser.parse(5, argv));
    REQUIRE(parser.get<int>("a0") == 4);
    REQUIRE(parser.get<int>("a1") == 1);
    REQUIRE(parser.get<int>("a2") == 2);
    REQUIRE(parser.check("a0"));
    REQUIRE(parser.check("a1"));
    REQUIRE(parser.check("a2"));
  }
  SECTION("print help") {
    ArgParser parser("test bin", "test bin description.");
    parser.addArgument("a0", "a0 description", false);
    parser.addArgument("a1", "a1 description", false);
    parser.addArgument("a2", "a2 description", true);
    parser.addArgument("a3", "a3 description", true);
    parser.printHelp();
  }
}

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
  SECTION("ls") {
    REQUIRE(FileSystem::mkdir("ls_folder/folder"));
    REQUIRE(FileSystem::writeFile("ls_folder/file1", "a"));
    REQUIRE(FileSystem::writeFile("ls_folder/file2", "b"));
    REQUIRE(FileSystem::writeFile("ls_folder/file3", "c"));
    auto ls = FileSystem::ls("ls_folder");
    REQUIRE(ls.size() == 6);
  }
}

TEST_CASE("Index", "[common][index]") {
  SECTION("Index2 arithmetic") {
    index2 ij(1,-3);
    REQUIRE(ij + index2(-7, 10) == index2(-6, 7));
    REQUIRE(ij - index2(-7, 10) == index2(8, -13));
    REQUIRE(ij + size2(7, 10) == index2(8, 7));
    REQUIRE(ij - size2(7, 10) == index2(-6, -13));
    REQUIRE(size2(7, 10) + ij == index2(8, 7));
    REQUIRE(size2(7, 10) - ij == index2(6, 13));
  }
  SECTION("Index2") {
    index2 a;
    index2 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  SECTION("Index2Range") {
    int cur = 0;
    for (auto index : Index2Range<i32>(10, 10)) {
      REQUIRE(cur % 10 == index.i);
      REQUIRE(cur / 10 == index.j);
      cur++;
    }
    REQUIRE(cur == 10 * 10);
  }
  SECTION("Index3 arithmetic") {
    index3 ij(1,-3, 0);
    REQUIRE(ij + index3(-7, 10, 1) == index3(-6, 7, 1));
    REQUIRE(ij - index3(-7, 10, 1) == index3(8, -13, -1));
    REQUIRE(ij + size3(7, 10, 3) == index3(8, 7, 3));
    REQUIRE(ij - size3(7, 10, 3) == index3(-6, -13, -3));
    REQUIRE(size3(7, 10, 5) + ij == index3(8, 7, 5));
    REQUIRE(size3(7, 10, 5) - ij == index3(6, 13, 5));
  }
  SECTION("Index3") {
    index3 a;
    index3 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  }
  SECTION("Index3Range") {
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