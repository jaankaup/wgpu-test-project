#version 450

layout(location = 0) in vec4 pos;

layout(location = 0) out vec4 final_color;

void main() {
    float color_a = pos.w;
    float color_b = 1.0 - pos.w;
    final_color = vec4(color_a, 0.0, color_b, 1.0);
}
