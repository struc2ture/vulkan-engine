#version 450

#extension GL_EXT_buffer_reference : require

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
	mat4 model;
	VertexBuffer vertexBuffer;
} pc;

layout (set = 0, binding = 0) uniform ShadowMapData
{
    mat4 lightSpaceTransform;

} shadowMapData;


void main()
{
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	vec4 position = vec4(v.position, 1.0f);
	gl_Position = shadowMapData.lightSpaceTransform * pc.model * position;
}