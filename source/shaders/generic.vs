#version 330 core

layout(location = 0) in vec4 vertex_pos;
layout(location = 1) in vec4 vertex_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 fragment_color;

void main()
{
	gl_Position = model*vec4(vertex_pos.x, vertex_pos.y, 0.0, 1.0);
	fragment_color = vec4(vertex_color.x,vertex_color.y,vertex_color.z,1.0);
}