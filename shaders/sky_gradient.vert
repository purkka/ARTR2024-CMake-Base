#version 460
#extension GL_GOOGLE_include_directive : enable
#include "shader_structures.glsl"
// -------------------------------------------------------

layout (location = 0) in vec3 aVertexPosition;

layout (location = 0) out vec3 vSphereCoords;

// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 0, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };

void main()
{
	vSphereCoords = aVertexPosition.xyz;
	vec4 position_cs = uboMatricesAndUserInput.mProjMatrix * uboMatricesAndUserInput.mViewMatrix * uboMatricesAndUserInput.mCamPos * vec4(aVertexPosition, 1.0);
	gl_Position = position_cs;
}

