layout (set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; // w for sun power
	vec4 sunlightColor;
	vec4 viewPos;
	
	vec4 lightPos[8];
	vec4 lightColor[8];
	//vec4 lightAmbient;
	//vec4 lightDiffuse;
	//vec4 lightSpecular;
	
	vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
	
	int lightsUsed;

} sceneData;

layout (set = 1, binding = 0) uniform GLTFMaterialData
{
	vec4 colorFactors;
	vec4 metal_rough_factors;
} materialData;

layout (set = 1, binding = 1) uniform sampler2D colorTex;
layout (set = 1, binding = 2) uniform sampler2D metalRoughTex;