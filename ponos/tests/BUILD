deps = ["@gtest//:main",
        "//ponos:ponos"]
copts = ["-Iexternal/gtest/include"]

cc_test(
    name = "queries",
    srcs = ["queries_tests.cpp"],
    copts = copts,
    deps = deps,
    tags = ["geometry"]
)

test_suite(
    name = "geometry",
    tags = ['geometry']
)