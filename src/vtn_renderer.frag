#version 450

layout(location = 0) in vec2 tex;
layout(location = 1) in vec4 nor;

layout(location = 0) out vec4 final_color;

layout(set = 1, binding = 0) uniform texture2D t_diffuse;
layout(set = 1, binding = 1) uniform texture2D s_diffuse;

void main() {
    final_color = texture(sampler2D(t_diffuse, s_diffuse), tex);
}
