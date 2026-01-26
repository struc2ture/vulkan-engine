#version 450

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outColor;

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

#define MAX_DEBUG_LINES 64

layout (set = 1, binding = 0) uniform LineData
{
    vec4 start[MAX_DEBUG_LINES];
    vec4 end[MAX_DEBUG_LINES];
    vec4 startColor[MAX_DEBUG_LINES];
    vec4 endColor[MAX_DEBUG_LINES];
    float thickness[MAX_DEBUG_LINES];

} lineData;

void main()
{
    int quadIndex = gl_VertexIndex % 6;
    int lineIndex = gl_VertexIndex / 6;

    vec4 lineStart_V = sceneData.viewproj * vec4(lineData.start[lineIndex].xyz, 1.0);
    lineStart_V /= lineStart_V.w;
    vec4 lineEnd_V = sceneData.viewproj * vec4(lineData.end[lineIndex].xyz, 1.0);
    lineEnd_V /= lineEnd_V.w;

    vec2 lineDir = normalize(lineEnd_V.xy - lineStart_V.xy);
    vec2 linePerp = vec2(-lineDir.y, lineDir.x);

    vec2 quadStartRight = lineStart_V.xy + linePerp * lineData.thickness[lineIndex] * 0.001;
    vec2 quadStartLeft = lineStart_V.xy - linePerp * lineData.thickness[lineIndex] * 0.001;
    vec2 quadEndRight = lineEnd_V.xy + linePerp * lineData.thickness[lineIndex] * 0.001;
    vec2 quadEndLeft = lineEnd_V.xy - linePerp * lineData.thickness[lineIndex] * 0.001;

    vec2 verts[] = { quadStartLeft, quadStartRight, quadEndLeft, quadEndRight };
    vec4 colors[] = { lineData.startColor[lineIndex], lineData.startColor[lineIndex], lineData.endColor[lineIndex], lineData.endColor[lineIndex] };
    int indices[] = { 0, 1, 2, 1, 2, 3 };
	
	gl_Position = vec4(verts[indices[quadIndex]], 0.0, 1.0);

    outColor = colors[indices[quadIndex]].rgb;
}