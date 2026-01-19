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
	
	// ambient
	vec3 ambient = sceneData.ambient.rgb;
	
	vec3 diffuseTotal = vec3(0.0);
	vec3 specularTotal = vec3(0.0);
	for (int i = 0; i < sceneData.lightsUsed; i++)
	{
		// diffuse
		vec3 norm = normalize(inNormal);
		vec3 lightDir = normalize(sceneData.lightPos[i].xyz - inFragPos);
		
		float diff = max(dot(norm, lightDir), 0.0);
		diffuseTotal += sceneData.lightColor[i].rgb * (diff * sceneData.diffuse.rgb);
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		//float spec = pow(max(dot(viewDir, reflectDir), 0.0), sceneData.shininess);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		specularTotal += sceneData.lightColor[i].rgb * (spec * sceneData.specular.rgb);
	}
	
	vec3 finalColor = (ambient + diffuseTotal + specularTotal) * albedo;
	outFragColor = vec4(finalColor, 1.0);
}
