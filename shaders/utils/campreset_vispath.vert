#version 460

layout (location = 0) in vec3 inPosition; 

layout(push_constant) uniform PushConstants {
	mat4 mViewProjMatrix;
	vec4 mColor;  // .a = point size
	vec4 mColor2;
	int  mVertexToHighlight;
	float pad1,pad2,pad3;
};

layout (location = 0) out vec3 outColor;

void main() {
	gl_Position = mViewProjMatrix * vec4(inPosition.xyz, 1.0);
	gl_PointSize = mColor.a;
	outColor = (gl_VertexIndex == mVertexToHighlight) ? mColor2.rgb : mColor.rgb;
}

