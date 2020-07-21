#version 450

layout(location = 0) in vec4 pos;

layout(location = 0) out vec4 final_color;

void main() {
    final_color = vec4(pos.w, 0.0, 0.0, 1.0);
}
