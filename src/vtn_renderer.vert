#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 tex;
layout(location = 2) in vec3 nor;

layout(location = 0) out vec3 pos_out;
layout(location = 1) out vec2 tex_out;
layout(location = 2) out vec3 nor_out;

layout(set=0, binding=0) uniform camerauniform {
    mat4 u_view_proj;
    vec3 camera_pos;
};

void main() {
    gl_Position = u_view_proj * vec4(pos, 1.0); 
    pos_out = pos;
    tex_out = tex; 
    nor_out = nor; 
}
