#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inColor;

layout (set = 1, binding = 0) uniform sampler2D iconTex;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec4 texel = texture(iconTex, inUV);
    if (texel.a < 0.1)
    {
        discard;
    }

    outFragColor = vec4(inColor * texel.rgb, 1.0);
}