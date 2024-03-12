#ifndef SHADER_STRUCTURES_GLSL

// ###### UNIFORMS AND PUSH CONSTANTS ###############
// Uniform buffer struct, containing camera matrices and user input:
// It is updated every frame.
struct matrices_and_user_input {
	// view matrix as returned from quake_camera
	mat4 mViewMatrix;
	// projection matrix as returned from quake_camera
	mat4 mProjMatrix;
	// transformation matrix which tranforms to camera's position
	mat4 mCamPos;
	// x = normal mapping strength, y = displacement strength, z = enable PN-triangles, w unused
	vec4 mUserInput;
};

struct PushConstants {
	mat4 mModelMatrix;
	int mMaterialIndex;
};

// ###### MATERIAL DATA ##################################
// Material data struct definition:
struct MaterialGpuData 
{
	vec4 mDiffuseReflectivity;
	vec4 mAmbientReflectivity;
	vec4 mSpecularReflectivity;
	vec4 mEmissiveColor;
	vec4 mTransparentColor;
	vec4 mReflectiveColor;
	vec4 mAlbedo;

	float mOpacity;
	float mBumpScaling;
	float mShininess;
	float mShininessStrength;

	float mRefractionIndex;
	float mReflectivity;
	float mMetallic;
	float mSmoothness;

	float mSheen;
	float mThickness;
	float mRoughness;
	float mAnisotropy;

	vec4 mAnisotropyRotation;
	vec4 mCustomData;

	int mDiffuseTexIndex;
	int mSpecularTexIndex;
	int mAmbientTexIndex;
	int mEmissiveTexIndex;
	int mHeightTexIndex;
	int mNormalsTexIndex;
	int mShininessTexIndex;
	int mOpacityTexIndex;
	int mDisplacementTexIndex;
	int mReflectionTexIndex;
	int mLightmapTexIndex;
	int mExtraTexIndex;

	vec4 mDiffuseTexOffsetTiling;
	vec4 mSpecularTexOffsetTiling;
	vec4 mAmbientTexOffsetTiling;
	vec4 mEmissiveTexOffsetTiling;
	vec4 mHeightTexOffsetTiling;
	vec4 mNormalsTexOffsetTiling;
	vec4 mShininessTexOffsetTiling;
	vec4 mOpacityTexOffsetTiling;
	vec4 mDisplacementTexOffsetTiling;
	vec4 mReflectionTexOffsetTiling;
	vec4 mLightmapTexOffsetTiling;
	vec4 mExtraTexOffsetTiling;
};

// ###### LIGHTSOURCE DATA ##################################
// Lightsource data struct definition in GPU-suitable format:
struct LightsourceGpuData
{
	/** Color of the light source. */
	vec4 mColor;
	/** Direction of the light source. */
	vec4 mDirection;
	/** Position of the light source. */
	vec4 mPosition;
	/** Angles, where the individual elements contain the following data: [0] cosine of halve outer cone angle, [1] cosine of halve inner cone angle, [2] falloff, [3] unused */
	vec4 mAnglesFalloff;
	/* Light source attenuation, where the individual elements contain the following data: [0] constant attenuation factor, [1] linear attenuation factor, [2] quadratic attenuation factor, [3], unused */
	vec4 mAttenuation;
	/** General information about the light source, where the individual elements contain the following data:[0] type of the light source */
	ivec4 mInfo;
};

// ###### TESSELLATION PATCH DATA #######################
// Patch data passed from tessellation control shader to tessellation evaluation shader:
struct PatchData
{
	vec3 b300;
	vec3 b030;
	vec3 b003;
	vec3 b210;
	vec3 b120;
	vec3 b021;
	vec3 b012;
	vec3 b102;
	vec3 b201;
	vec3 b111;

	vec3 n200;
	vec3 n020;
	vec3 n002;
	vec3 n110;
	vec3 n011;
	vec3 n101;
};

#define SHADER_STRUCTURES_GLSL 1
#endif

