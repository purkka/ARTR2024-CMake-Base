#version 430 core

layout(push_constant) uniform PushConstants {
	mat4 pvmtMatrix;
	vec4 uColor;
};

layout (location = 0) in vec4 aVertexPosition; 

void main()
{	
	gl_Position = pvmtMatrix * aVertexPosition;
}

