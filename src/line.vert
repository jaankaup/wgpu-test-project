#version 450

layout(location = 0) in vec2 pos;

layout(location = 0) out vec2 pos_out;

layout(set=0, binding=0) uniform camerauniform {
    mat4 u_view_proj;
    vec3 camera_pos;
};

void main() {
    gl_Position = u_view_proj * vec4(pos.x, pos.y, 0.0, 1.0); 
    pos_out = pos;
}
