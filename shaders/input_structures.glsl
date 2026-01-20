layout (set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; // w for sun power
	vec4 sunlightColor;
	vec4 viewPos;
	
	//vec4 lightPos[8];
	//vec4 lightColor[8];
	//vec4 lightAmbient;
	//vec4 lightDiffuse;
	//vec4 lightSpecular;
	
	vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
	    
    int dirsUsed;
    int pointsUsed;
    int spotsUsed;

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

layout (set = 1, binding = 0) uniform GLTFMaterialData
{
	vec4 colorFactors;
	vec4 metal_rough_factors;

} materialData;

layout (set = 1, binding = 1) uniform sampler2D colorTex;
layout (set = 1, binding = 2) uniform sampler2D metalRoughTex;