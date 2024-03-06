#version 430 core

layout(push_constant) uniform PushConstants {
	mat4 pvmtMatrix;
	vec4 uColor;
};

// ------------- output-color of fragment ------------
layout (location = 0) out vec4 oFragColor;

void main()
{
	oFragColor = uColor;
}

