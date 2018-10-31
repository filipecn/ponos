new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "ext/BUILD.gtest",
    strip_prefix = "googletest-release-1.7.0",
)

new_http_archive(
    name = "glfw",
    url = "https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip",
    build_file = "ext/BUILD.glfw",
    strip_prefix = "glfw-3.2.1"
)

new_http_archive(
    name = "glfwWin32",
    url = "https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.bin.WIN32.zip",
    build_file = "ext/BUILD.glfwWin32",
    strip_prefix = "glfw-3.2.1.bin.WIN32"
)

new_git_repository(
    name = "nanogui",
    remote = "https://github.com/filipecn/nanogui.git",
    commit = "c9e2f6160aa23dc267f4eabe3ded174d319cfba3",
    build_file = "ext/BUILD.nanogui"
)

new_git_repository(
    name = "nanovg",
    remote = "https://github.com/memononen/nanovg.git",
    commit = "f4069e6a1ad5da430fb0a9c57476d5ddc2ff89b2",
    build_file = "ext/BUILD.nanovg"
)

new_http_archive(
    name = "eigen",
    url = "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.zip",
    build_file = "ext/BUILD.eigen",
    strip_prefix = "eigen-git-mirror-3.3.5"
)