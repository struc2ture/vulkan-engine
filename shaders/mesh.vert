#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outFragPos;
layout (location = 4) out mat3 outTBN;

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
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main()
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);
	
	gl_Position = sceneData.viewproj * PushConstants.render_matrix * position;
	
	mat3 normalMat = mat3(transpose(inverse(PushConstants.render_matrix)));
	
	outNormal = normalMat * v.normal;
	outColor = v.color.rgb * materialData.colorFactors.rgb;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
	outFragPos = vec3(PushConstants.render_matrix * position);
	
	vec3 T = normalize(vec3(normalMat * v.tangent.xyz));
	vec3 N = normalize(vec3(normalMat * v.normal));
	vec3 B = cross(N, T) * v.tangent.w;
	outTBN = mat3(T, B, N);
}