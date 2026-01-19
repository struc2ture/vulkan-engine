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
	vec3 norm = normalize(inNormal);
	
	
	vec3 finalLight = vec3(0.0);
	
	// ambient
	vec3 ambient = sceneData.ambient.rgb;
	finalLight += ambient;
	
	// directional lights
	/*
	{
		vec3 lightDirection = vec3(-0.2, -1.0, -0.3);
		vec3 lightColor = vec3(1.0);
		
		vec3 lightDir = -lightDirection;
		
		// diffuse
		float diff = max(dot(norm, lightDir), 0.0);
		finalLight += lightColor.rgb * (diff * sceneData.diffuse.rgb);
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		//float spec = pow(max(dot(viewDir, reflectDir), 0.0), sceneData.shininess);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess);
		finalLight += lightColor.rgb * (spec * sceneData.specular.rgb);
	}
	*/
	
	// point lights
	for (int i = 0; i < lightsData.lightsUsed; i++)
	{
		// attenuation
		float attenuationConstant = 1.0;
		float attenuationLinear = 0.7;
		float attenuationQuadratic = 1.8;
		
		float distance = length(lightsData.lightPos[i].xyz - inFragPos);
		float attenuation = 1.0 / (attenuationConstant + attenuationLinear * distance + attenuationQuadratic * distance * distance);
		
		// diffuse
		vec3 lightDir = normalize(lightsData.lightPos[i].xyz - inFragPos);
		
		float diff = max(dot(norm, lightDir), 0.0) * attenuation;
		finalLight += lightsData.lightColor[i].rgb * (diff * sceneData.diffuse.rgb);
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		//float spec = pow(max(dot(viewDir, reflectDir), 0.0), sceneData.shininess);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess) * attenuation;
		finalLight += lightsData.lightColor[i].rgb * (spec * sceneData.specular.rgb);
	}
	
	// spotlights
	/*
	{
		vec3 lightPos = vec3(0.0, 1.0, 3.0);
		vec3 spotDir = normalize(-lightPos);
		vec3 lightColor = vec3(1.0);
		float cutOff = cos(radians(5));
		float outerCutoff = cos(radians(8));
		
		// attenuation
		float attenuationConstant = 1.0;
		float attenuationLinear = 0.09;
		float attenuationQuadratic = 0.032;
		
		float distance = length(lightPos.xyz - inFragPos);
		float attenuation = 1.0 / (attenuationConstant + attenuationLinear * distance + attenuationQuadratic * distance * distance);

		vec3 lightDir = normalize(lightPos.xyz - inFragPos);
		
		float theta = dot(lightDir, normalize(-spotDir));
		float epsilon = cutOff - outerCutoff;
		float intensity = clamp((theta - outerCutoff) / epsilon, 0.0, 1.0);

		// diffuse
		float diff = max(dot(norm, lightDir), 0.0) * intensity * attenuation;
		finalLight += lightColor.rgb * (diff * sceneData.diffuse.rgb);
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		//float spec = pow(max(dot(viewDir, reflectDir), 0.0), sceneData.shininess);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess) * intensity * attenuation;
		finalLight += lightColor.rgb * (spec * sceneData.specular.rgb);
	}
	*/
	
	vec3 finalColor = finalLight * albedo;
	
	outFragColor = vec4(finalColor, 1.0);
}
