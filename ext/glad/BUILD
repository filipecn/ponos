licenses(["notice"])

cc_library(
    name = "glad",
    srcs = ["glad.c"],
    hdrs = ["include/glad/glad.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":khr"],
)

cc_library(
    name = "khr",
    hdrs = ["include/KHR/khrplatform.h"],
    visibility = ["//visibility:public"],
)