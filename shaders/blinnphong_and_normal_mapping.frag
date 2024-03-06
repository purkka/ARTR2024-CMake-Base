#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_post_depth_coverage : enable
#include "lightsource_limits.h"
#include "shader_structures.glsl"
// -------------------------------------------------------


// ###### MATERIAL DATA ##################################
// The actual material buffer (of type MaterialGpuData):
// It is bound to descriptor set at index 0 and
// within the descriptor set, to binding location 0
layout(set = 0, binding = 0) buffer Material
{
	MaterialGpuData materials[];
} materialsBuffer;

// Array of samplers containing all the material's images:
// These samplers are referenced from materials by
// index, namely by all those m*TexIndex members.
layout(set = 0, binding = 1) uniform sampler2D textures[];
// -------------------------------------------------------

// ###### PIPELINE INPUT DATA ############################
// Unique push constants per draw call (You can think of
// these like single uniforms in OpenGL):
layout(push_constant) uniform PushConstantsBlock { PushConstants pushConstants; };

// Uniform buffer "uboMatricesAndUserInput", containing camera matrices and user input
layout (set = 1, binding = 0) uniform UniformBlock { matrices_and_user_input uboMatricesAndUserInput; };

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) uniform LightsourceData
{
	// x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices
	uvec4 mRangesAmbientDirectional;
	// x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices
	uvec4 mRangesPointSpot;
	// Contains all the data of all the active light sources
	LightsourceGpuData mLightData[MAX_NUMBER_OF_LIGHTSOURCES];
} uboLights;
// -------------------------------------------------------

// ###### FRAG INPUT #####################################
layout (location = 0) in VertexData
{
	vec3 positionVS;  // interpolated vertex position in view-space
	vec2 texCoords;   // texture coordinates
	vec3 normalVS;    // interpolated vertex normal in view-space
	// TODO Task 2: Receive whatever data you have passed from previous shader stages!
} fs_in;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragColor;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################
vec4 sample_from_diffuse_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_height_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mHeightTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mHeightTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_normals_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mNormalsTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mNormalsTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

// Re-orthogonalizes the first vector w.r.t. the second vector (Gram-Schmidt process)
vec3 re_orthogonalize(vec3 first, vec3 second)
{
	return normalize(first - dot(first, second) * second);
}

// Calculates the light attenuation dividend for the given attenuation vector.
// @param atten attenuation data
// @param dist  distance
// @param dist2 squared distance
float calc_attenuation(vec4 atten, float dist, float dist2)
{
	return atten[0] + atten[1] * dist + atten[2] * dist2;
}

// Calculates the diffuse and specular illumination contribution for the given
// parameters according to the Blinn-Phong lighting model.
// All parameters must be normalized.
vec3 calc_blinn_phong_contribution(vec3 toLight, vec3 toEye, vec3 normal, vec3 diffFactor, vec3 specFactor, float specShininess)
{
	float nDotL = max(0.0, dot(normal, toLight)); // lambertian coefficient
	vec3 h = normalize(toLight + toEye);
	float nDotH = max(0.0, dot(normal, h));
	float specPower = pow(nDotH, specShininess);

	vec3 diffuse = diffFactor * nDotL; // component-wise product
	vec3 specular = specFactor * specPower;

	return diffuse + specular;
}

// Calculates the diffuse and specular illumination contribution for all the light sources.
// All calculations are performed in view space
vec3 calc_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 diff, vec3 spec, float shini)
{
	vec3 diffAndSpec = vec3(0.0, 0.0, 0.0);

	// Calculate shading in view space since all light parameters are passed to the shader in view space
	vec3 eyePosVS = vec3(0.0, 0.0, 0.0);
	vec3 toEyeNrmVS = normalize(eyePosVS - posVS);

	// Directional lights:
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		diffAndSpec += dirLightIntensity * calc_blinn_phong_contribution(toLightDirVS, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// Point lights:
	// TODO Task 2: Implement Blinn-Phong illumination for all the point lights contained within uboLights!

	return diffAndSpec;
}
// -------------------------------------------------------

// ###### FRAGMENT SHADER MAIN #############################
void main()
{
	vec3 positionVS = fs_in.positionVS;
	// TODO Task 2: Compute a normal using the normal map! Feel free to use sample_from_normals_texture() 
	//              to sample from the appropriate normal map texture, as specified by the current material.
	vec3 normalVS = fs_in.normalVS;

	// Sample the diffuse color:
	vec3 diffTexColor = sample_from_diffuse_texture().rgb;

	int matIndex = pushConstants.mMaterialIndex;

	// Initialize all the colors:
	vec3 ambient = materialsBuffer.materials[matIndex].mAmbientReflectivity.rgb * diffTexColor;
	vec3 emissive = materialsBuffer.materials[matIndex].mEmissiveColor.rgb;
	vec3 diff = materialsBuffer.materials[matIndex].mDiffuseReflectivity.rgb * diffTexColor;
	vec3 spec = materialsBuffer.materials[matIndex].mSpecularReflectivity.rgb * sample_from_specular_texture().r;
	float shininess = materialsBuffer.materials[matIndex].mShininess;

	// Calculate ambient illumination:
	vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
	for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
		ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
	}

	// Calculate diffuse and specular illumination from all light sources:
	vec3 diffAndSpecIllumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess);

	// Add all together:
	oFragColor = vec4(ambientIllumination + emissive + diffAndSpecIllumination, 1.0);

	// TODO Task 6, TODO Bonus Task 2:
	//  - Read roughness and metallic values from textures and pass them on to the lighting subpass
	//  - You can use the provided functions sample_roughness() and sample_metallic() to read from the textures
	//  - It is up to you, how you pass on the values (but try to be not too wasteful with resources)
}
// -------------------------------------------------------

