#version 440 core
layout(location = 0) out vec4 fragColor;

layout(location = 4) uniform vec4 color;

void main() { fragColor = color; }