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

COMMONDEPS = ["//aergia",
            "//deps/glad:glad"]
LINKOPTS_WIN = ["kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib"]
LINKOPTS = ["-ldl", "-lGL", "-lGLU", "-lX11", "-lXrandr", "-lXxf86vm", "-lXinerama","-lXcursor", "-lpthread"]

cc_library (
    name = "screen",
    hdrs = ["nanogui_screen.h"],
    srcs = ["nanogui_screen.cpp"],
    includes = ["."],
    deps = ["//aergia",
            "@nanogui//:nanogui",
            #"//deps/glad:glad",
            #"@glfw//:glfw",
            ],
    linkopts = select({
        ":windows": LINKOPTS_WIN,
        ":linux": LINKOPTS,"//conditions:default": []}),
)

cc_binary(
    name = "nanogui",
    srcs = ["nanogui_example.cpp"],
    deps = [":screen"]
)