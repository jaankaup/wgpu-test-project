#version 450

layout(location = 0) in vec4 pos;
layout(location = 1) in vec2 tex;
layout(location = 2) in vec4 nor;

layout(location = 0) out vec4 tex_out;
layout(location = 1) out vec4 nor_out;

layout(set=0, binding=0) uniform camerauniform {
    mat4 u_view_proj;
};

void main() {
    gl_position = u_view_proj * pos; 
}
