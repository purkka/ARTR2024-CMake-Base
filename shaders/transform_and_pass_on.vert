#version 460
#extension GL_GOOGLE_include_directive : enable
#include "shader_structures.glsl"
// -------------------------------------------------------

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
// Several vertex attributes (These are the buffers passed
// to command_buffer_t::draw_indexed in the same order):
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;
// TODO Task 1: Declare from which input locations to receive tangent and bitangent data!

// Unique push constants per draw call (You can think of
// these like single uniforms in OpenGL):
layout(push_constant) uniform PushConstantsBlock { PushConstants pushConstants; };

// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 1, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };
// -------------------------------------------------------

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	vec3 positionVS;
	vec2 texCoords;
	vec3 normalVS;
	// TODO Task 2: Pass whatever data makes sense for normal mapping to subsequent shader stages!
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	mat4 mMatrix = pushConstants.mModelMatrix;
	mat4 vMatrix = uboMatricesAndUserInput.mViewMatrix;
	mat4 pMatrix = uboMatricesAndUserInput.mProjMatrix;
	mat4 vmMatrix = vMatrix * mMatrix;
	mat4 pvmMatrix = pMatrix * vmMatrix;

	vec4 positionOS  = vec4(aPosition, 1.0);
	vec4 positionVS  = vmMatrix * positionOS;
	vec4 positionCS  = pMatrix * positionVS;
	vec3 normalOS    = normalize(aNormal);
	vec3 normalVS    = normalize(mat3(inverse(transpose(vmMatrix))) * normalOS);

	v_out.positionVS  = positionVS.xyz;
	v_out.texCoords   = aTexCoords;
	v_out.normalVS    = normalVS;

	gl_Position = positionCS;
}
// -------------------------------------------------------

