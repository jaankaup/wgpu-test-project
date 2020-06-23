#version 450

layout(location = 0) in vec4 pos;
layout(location = 0) out vec4 pos_out;

void main() {
    pos_out = pos; 
}
