#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inFragPos;
layout (location = 4) in vec4 inFragPosLightSpace;
layout (location = 5) in mat3 inTBN;

layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	mat4 lightSpaceTransform;
	vec4 ambient;
	vec4 viewPos;
    vec4 cameraRight;
    vec4 cameraUp;
	vec4 aspect;

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
	vec4 bypassLight;

} materialData;

layout (set = 1, binding = 1) uniform sampler2D diffuseTex;
layout (set = 1, binding = 2) uniform sampler2D specularTex;
layout (set = 1, binding = 3) uniform sampler2D emissionTex;
layout (set = 1, binding = 4) uniform sampler2D normalTex;
layout (set = 1, binding = 5) uniform sampler2D parallaxTex;

layout (set = 2, binding = 0) uniform sampler2D shadowMapDepth;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    float closestDepth = texture(shadowMapDepth, projCoords.xy * 0.5 + 0.5).r; 
    float currentDepth = projCoords.z;
	// Here we have a maximum bias of 0.05 and a minimum of 0.005 based on the surface's normal and light direction.
	// float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
	float bias = 0.005;
    float shadow = (currentDepth - bias) > closestDepth  ? 1.0 : 0.0;
	if (projCoords.z > 1.0) shadow = 0.0;
    return shadow;
}

float Test(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMapDepth, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    float shadow = closestDepth;
    return shadow;
}

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
	
	
	// ambient
	vec3 ambient = sceneData.ambient.rgb * diffuseTexel;

	vec3 diffSpecLight = vec3(0.0);
	
	// directional lights
	for (int i = 0; i < lightsData.dirsUsed; i++)
	{
		vec3 lightDirection = lightsData.dirDir[i].xyz;
		vec3 lightColor = lightsData.dirColor[i].rgb;
		float lightPower = lightsData.dirDir[i].w;
		
		vec3 lightDir = -lightDirection;
		
		// diffuse
		float diff = max(dot(norm, lightDir), 0.0);
		
		diffSpecLight += lightColor * lightPower * diff * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		
		diffSpecLight += lightColor * lightPower * spec * materialData.specular.rgb * specularTexel;
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
		
		diffSpecLight += lightColor * diff * attenuation * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		
		diffSpecLight += lightColor * spec * attenuation * materialData.specular.rgb * specularTexel;
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
		diffSpecLight += lightColor * diff * intensity * attenuation * materialData.diffuse.rgb * diffuseTexel;
		
		// specular
		vec3 viewDir = normalize(sceneData.viewPos.xyz - inFragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float shininess = materialData.specular.a;
		float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
		diffSpecLight += lightColor * spec * intensity * attenuation * materialData.specular.rgb * specularTexel;
	}
	
	// emission
	vec3 emission = texture(emissionTex, inUV).rgb * materialData.emission.rgb;

	float shadow = ShadowCalculation(inFragPosLightSpace, norm, lightsData.dirDir[0].xyz);

	vec3 litColor = ambient + diffSpecLight * (1.0 - shadow) + emission;
	vec3 bypassedColor = materialData.diffuse.rgb * diffuseTexel;

	vec3 color = mix(litColor, bypassedColor, materialData.bypassLight.r);
	outFragColor = vec4(color, diffuseT.a * materialData.diffuse.a);
}
