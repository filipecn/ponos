#version 440 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 fColor;

layout(location = 2) uniform mat4 model;
layout(location = 3) uniform mat4 view;
layout(location = 4) uniform mat4 projection;

void main() {
  gl_Position = projection * view * model * vec4(position, 1.0);
  gl_PointSize = 5;
  fColor = color;
}