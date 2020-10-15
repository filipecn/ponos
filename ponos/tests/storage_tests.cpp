#include <catch2/catch.hpp>

#include <ponos/ponos.h>

using namespace ponos;

TEST_CASE("Array1", "[storage][array]") {
  SECTION("Constructors") {
    {
      Array1<vec2> a(10);
      REQUIRE(a.size() == 10u);
      REQUIRE(a.memorySize() == 10 * sizeof(vec2));
      for (u64 i = 0; i < a.size(); ++i)
        a[i] = vec2(i, i * 2);
      Array1<vec2> b = a;
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == b[i]);
    } //
    {
      std::vector<Array1<int>> v;
      v.emplace_back(10);
      v.emplace_back(10);
      v.emplace_back(10);
      for (int i = 0; i < 3; i++)
        for (u64 j = 0; j < v[i].size(); ++j)
          v[i][j] = j * 10;
      std::vector<Array1<int>> vv = v;
      for (int i = 0; i < 3; i++)
        for (u64 j = 0; j < v[i].size(); ++j)
          REQUIRE(vv[i][j] == j * 10);
    } //
    {
      Array1<int> a = std::move(Array1<int>(10));
      auto b(Array1<int>(10));
    } //
    {
      std::vector<int> data = {1, 2, 3, 4, 5, 6};
      Array1<int> a = data;
      REQUIRE(a.size() == 6);
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == data[i]);
    } //
    {
      Array1<int> a = {1, 2, 3};
      REQUIRE(a.size() == 3);
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == i + 1);
    }
  }//
  SECTION("Operators") {
    Array1<int> a(10);
    a = 3;
    int count = 0;
    for (u64 i = 0; i < a.size(); ++i)
      REQUIRE(a[i] == 3);
    for (const auto &e : a) {
      REQUIRE(e.value == 3);
      count++;
    }
    std::cerr << a << std::endl;
    REQUIRE(count == 10);
  }//
  SECTION("Array1-iterator") {
    Array1<vec2> a(10);
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      REQUIRE(e.value == vec2(1, 2));
    }
    REQUIRE(count == 10);
  }//
  SECTION("Const Array1-iterator") {
    Array1<vec2> a(10);
    a = vec2(1, 2);
    auto f = [](const Array1<vec2> &array) {
      for (const auto &d : array)
        REQUIRE(d.value == vec2(1, 2));
    };
    f(a);
  }//
}

TEST_CASE("Array2", "[storage][array]") {
  SECTION("Constructors") {
    {
      Array2<vec2> a(size2(10, 10));
      REQUIRE(a.pitch() == 10 * sizeof(vec2));
      REQUIRE(a.size() == size2(10, 10));
      REQUIRE(a.memorySize() == 10 * 10 * sizeof(vec2));
      for (index2 ij : Index2Range<i32>(a.size()))
        a[ij] = vec2(ij.i, ij.j);
      Array2<vec2> b = a;
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == b[ij]);
    }
    {
      std::vector<Array2<int>> v;
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      for (int i = 0; i < 3; i++)
        for (index2 ij : Index2Range<i32>(v[i].size()))
          v[i][ij] = ij.i * 10 + ij.j;
      std::vector<Array2<int>> vv = v;
      for (int i = 0; i < 3; i++)
        for (index2 ij : Index2Range<i32>(v[i].size()))
          REQUIRE(vv[i][ij] == ij.i * 10 + ij.j);
    }
    {
      Array2<int> a = std::move(Array2<int>(size2(10, 10)));
      Array2<int> b(Array2<int>(size2(10, 10)));
    }
    {
      std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
      Array2<int> a = data;
      REQUIRE(a.size() == size2(3, 2));
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == data[ij.j][ij.i]);
    }
    {
      Array2<int> a = {{1, 2, 3}, {11, 12, 13}};
      REQUIRE(a.size() == size2(3, 2));
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == ij.j * 10 + ij.i + 1);
    }
  }//
  SECTION("Operators") {
    {
      Array2<int> a(size2(10, 10));
      a = 3;
      int count = 0;
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == 3);
      for (const auto &e : a) {
        REQUIRE(e.value == 3);
        count++;
      }
      std::cerr << a << std::endl;
      REQUIRE(count == 10 * 10);
    }
  }//
  SECTION("Array2-iterator") {
    Array2<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      REQUIRE(e.value == vec2(1, 2));
    }
    REQUIRE(count == 100);
  }//
  SECTION("Const Array2-iterator") {
    Array2<vec2> a(size2(10, 10));
    a = vec2(1, 2);
    auto f = [](const Array2<vec2> &array) {
      for (const auto &d : array)
        REQUIRE(d.value == vec2(1, 2));
    };
    f(a);
  }//
}

TEST_CASE("AOS", "[storage][aos]") {
  SECTION("Sanity Checks") {
    AoS aos;
    REQUIRE(aos.pushField<vec3>("vec3") == 0);
    REQUIRE(aos.pushField<f32>("f32") == 1);
    REQUIRE(aos.pushField<int>("int") == 2);
    REQUIRE(aos.fieldName(0) == "vec3");
    REQUIRE(aos.fieldName(1) == "f32");
    REQUIRE(aos.fieldName(2) == "int");
    REQUIRE(aos.size() == 0);
    // check fields
    auto fields = aos.fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "vec3");
    REQUIRE(fields[0].size == sizeof(vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
    aos.resize(4);
    REQUIRE(aos.size() == 4);
    REQUIRE(aos.stride() == sizeof(vec3) + sizeof(f32) + sizeof(int));
    REQUIRE(aos.sizeOf("vec3") == sizeof(vec3));
    REQUIRE(aos.sizeOf("f32") == sizeof(f32));
    REQUIRE(aos.sizeOf("int") == sizeof(int));
    REQUIRE(aos.offsetOf("vec3") == 0);
    REQUIRE(aos.offsetOf("f32") == sizeof(vec3));
    REQUIRE(aos.offsetOf("int") == sizeof(vec3) + sizeof(f32));
    REQUIRE(aos.memorySizeInBytes() == aos.stride() * 4);
    for (int i = 0; i < 4; ++i) {
      aos.valueAt<vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      aos.valueAt<f32>(1, i) = 1.f * i;
      aos.valueAt<int>(2, i) = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
      REQUIRE(aos.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(aos.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(aos.valueAt<int>(2, i) == i + 1);
    }
  }//
  SECTION("Accessors") {
    AoS aos;
    aos.pushField<vec3>("vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    aos.resize(4);
    auto vec3_field = aos.field<vec3>("vec3");
    auto f32_field = aos.field<f32>("f32");
    auto int_field = aos.field<int>("int");
    for (int i = 0; i < 4; ++i) {
      vec3_field[i] = {1.f + i, 2.f + i, 3.f + i};
      f32_field[i] = 1.f * i;
      int_field[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
      REQUIRE(aos.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(aos.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(aos.valueAt<int>(2, i) == i + 1);
      REQUIRE(vec3_field[i] == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(f32_field[i] == 1.f * i);
      REQUIRE(int_field[i] == i + 1);
    }
  }
}