#version 450

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inFragPos;

layout (location = 0) out vec4 outFragColor;

void main()
{
	// Alpha cut-out
	vec4 texel =  texture(colorTex, inUV);
	if (texel.a < 0.1)
	{
		discard;
	}
	
	vec3 albedo = inColor * texel.rgb;
	outFragColor = vec4(albedo, 1.0);
}
