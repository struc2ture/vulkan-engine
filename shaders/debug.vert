#version 450

#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outColor;

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	vec4 tangent; // w - bitangent handedness
};

layout (buffer_reference, std430) readonly buffer VertexBuffer
{
	Vertex vertices[];
};

layout (push_constant) uniform constants
{
	vec4 billboardPos;
    vec4 billboardSize;
    vec4 billboardColor;
	VertexBuffer vertexBuffer;
} PushConstants;

layout (set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambient;
	vec4 viewPos;
    vec4 cameraRight;
    vec4 cameraUp;
	vec4 aspect;

} sceneData;

void main()
{
    vec3 particleCenter = PushConstants.billboardPos.xyz;
	Vertex vert = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	// vec3 position = particleCenter +
    //     sceneData.cameraRight.xyz * vert.position.x * PushConstants.billboardSize.x +
    //     sceneData.cameraUp.xyz * vert.position.y * PushConstants.billboardSize.y;
	
	gl_Position = sceneData.viewproj * vec4(particleCenter, 1.0);
	gl_Position /= gl_Position.w;

	gl_Position.xy += vert.position.xy * PushConstants.billboardSize.xy * vec2(1.0, sceneData.aspect.x);

    outUV.x = vert.uv_x;
	outUV.y = vert.uv_y;
    outColor = PushConstants.billboardColor.rgb;
}