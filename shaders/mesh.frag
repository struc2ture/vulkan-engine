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
	for (int i = 0; i < lightsData.dirsUsed; i++)
	{
		vec3 lightDirection = lightsData.dirDir[i].xyz;
		vec3 lightColor = lightsData.dirColor[i].rgb;
		float lightPower = lightsData.dirDir[i].w;
		
		vec3 lightDir = -lightDirection;
		
		// diffuse
		float diff = max(dot(norm, lightDir), 0.0);
		
		finalLight += lightColor * lightPower * diff * sceneData.diffuse.rgb;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess);
		
		finalLight += lightColor * lightPower * spec * sceneData.specular.rgb;
	}
	
	// point lights
	for (int i = 0; i < lightsData.pointsUsed; i++)
	{
		vec3 lightPos = lightsData.pointPos[i].xyz;
		vec3 lightColor = lightsData.pointColor[i].rgb;
		float attenuationConstant = 1.0;
		float attenuationLinear = lightsData.pointAtten[i].x;
		float attenuationQuadratic = lightsData.pointAtten[i].y;

		// attenuation
		float distance = length(lightPos - inFragPos);
		float attenuation = 1.0 / (attenuationConstant + attenuationLinear * distance + attenuationQuadratic * distance * distance);
		
		// diffuse
		vec3 lightDir = normalize(lightPos - inFragPos);
		
		float diff = max(dot(norm, lightDir), 0.0);
		
		finalLight += lightColor * diff * attenuation * sceneData.diffuse.rgb;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess);
		
		finalLight += lightColor * spec * attenuation * sceneData.specular.rgb;
	}
	
	// spotlights
	for (int i = 0; i < lightsData.spotsUsed; i++)
	{
		vec3 lightPos = lightsData.spotPos[i].xyz;
		vec3 lightColor = lightsData.spotColor[i].rgb;
		vec3 spotDir = normalize(lightsData.spotDir[i].xyz);
		float attenuationConstant = 1.0;
		float attenuationLinear = lightsData.spotAttenCutoff[i].x;
		float attenuationQuadratic = lightsData.spotAttenCutoff[i].y;
		float cutOff = lightsData.spotAttenCutoff[i].z;
		float outerCutoff = lightsData.spotAttenCutoff[i].w;
		
		// attenuation
		float distance = length(lightPos - inFragPos);
		float attenuation = 1.0 / (attenuationConstant + attenuationLinear * distance + attenuationQuadratic * distance * distance);

		vec3 lightDir = normalize(lightPos - inFragPos);
		
		float theta = dot(lightDir, normalize(-spotDir));
		float epsilon = cutOff - outerCutoff;
		float intensity = clamp((theta - outerCutoff) / epsilon, 0.0, 1.0);

		// diffuse
		float diff = max(dot(norm, lightDir), 0.0);
		finalLight += lightColor * diff * intensity * attenuation * sceneData.diffuse.rgb;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(norm, halfwayDir), 0.0), sceneData.shininess);
		finalLight += lightColor * spec * intensity * attenuation * sceneData.specular.rgb;
	}
	
	vec3 finalColor = finalLight * albedo;
	
	outFragColor = vec4(finalColor, 1.0);
}
