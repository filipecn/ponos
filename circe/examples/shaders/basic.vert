#version 440 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 fNormal;
out vec3 fPosition;

//layout(std140) uniform Transforms {
layout (location = 0) uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
//};

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fPosition = vec3(model * vec4(position, 1.0));
    fNormal = mat3(transpose(inverse(model))) * normal;
}