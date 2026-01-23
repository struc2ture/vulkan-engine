#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inFragPos;
layout (location = 4) in mat3 inTBN;

layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambient;
	vec4 viewPos;

} sceneData;

#define MAX_LIGHTS 64
layout (set = 0, binding = 1) uniform LightsData
{
    // directional lights
    vec4 dirDir[MAX_LIGHTS]; // w for power
    vec4 dirColor[MAX_LIGHTS];

    // point lights
    vec4 pointPos[MAX_LIGHTS];
    vec4 pointColor[MAX_LIGHTS];
    vec4 pointAtten[MAX_LIGHTS]; // x - linear, y - quad

    // spotlights
    vec4 spotPos[MAX_LIGHTS];
    vec4 spotDir[MAX_LIGHTS];
    vec4 spotColor[MAX_LIGHTS];
    vec4 spotAttenCutoff[MAX_LIGHTS]; // x - linear, y - quad, z - cutoff, w - outer cutoff
    
    int dirsUsed;
	int pointsUsed;
	int spotsUsed;

} lightsData;

layout (set = 1, binding = 0) uniform MaterialData
{
	vec4 diffuse;
	vec4 specular; // a - shininess
	vec4 emission; // a - unused

} materialData;

layout (set = 1, binding = 1) uniform sampler2D diffuseTex;
layout (set = 1, binding = 2) uniform sampler2D specularTex;
layout (set = 1, binding = 3) uniform sampler2D emissionTex;
layout (set = 1, binding = 4) uniform sampler2D normalTex;
layout (set = 1, binding = 5) uniform sampler2D parallaxTex;

void main()
{
	vec4 diffuseT =  texture(diffuseTex, inUV);
	
	// Alpha cut-out
	if (diffuseT.a < 0.1)
	{
		discard;
	}
	
	vec3 diffuseTexel = inColor * diffuseT.rgb;
	vec3 specularTexel =  texture(specularTex, inUV).rgb;
	vec3 emissionTexel =  texture(emissionTex, inUV).rgb;
	
	vec3 norm = normalize(inNormal);
	
	norm = texture(normalTex, inUV).rgb;
	norm = normalize(norm * 2.0 - 1.0);
	norm = normalize(inTBN * norm);
	
	vec3 finalLight = vec3(0.0);
	
	// ambient
	vec3 ambient = sceneData.ambient.rgb * diffuseTexel;
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
		
		finalLight += lightColor * lightPower * diff * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		
		finalLight += lightColor * lightPower * spec * materialData.specular.rgb * specularTexel;
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
		
		finalLight += lightColor * diff * attenuation * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		
		finalLight += lightColor * spec * attenuation * materialData.specular.rgb * specularTexel;
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
		finalLight += lightColor * diff * intensity * attenuation * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		finalLight += lightColor * spec * intensity * attenuation * materialData.specular.rgb * specularTexel;
	}
	
	// emission
	{
		vec3 emission = texture(emissionTex, inUV).rgb;
		finalLight += emission * materialData.emission.rgb;
	}
	
	vec3 finalColor = finalLight;
	
	outFragColor = vec4(finalColor, 1.0);
}
