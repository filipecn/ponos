config_setting(
  name = "linux",
  values = { "cpu": "k8" },
)
config_setting(
  name = "osx",
  values = { "cpu": "darwin" },
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"}
)

COMMON_DEPS = ["//circe",
            "//ext/glad:glad",
            "@glfw//:glfw"]
LINKOPTS_WIN = ["kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib"]
LINKOPTS = ["-ldl", "-lGL", "-lGLU", "-lX11", "-lXrandr", "-lXxf86vm", "-lXinerama","-lXcursor", "-lpthread"]

cc_binary(
    name = "instances",
    srcs = ["instances.cpp"],
    deps = COMMON_DEPS,
    linkopts = select({
      ":windows": LINKOPTS_WIN,
      ":linux": LINKOPTS,"//conditions:default": []}),
)

cc_binary(
    name = "hello",
    srcs = ["hello_circe.cpp"],
    deps = COMMON_DEPS,
    data = ["//circe/examples/shaders:basic.vs",
    "//circe/examples/shaders:basic.fs"],
    linkopts = select({
      ":windows": LINKOPTS_WIN,
      ":linux": LINKOPTS,"//conditions:default": []}),
)