// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>


#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

static inline float rand_float()
{
    float rand = (float)std::rand() / (float)RAND_MAX;
    return rand;
}

struct AllocatedImage
{
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex
{
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
    glm::vec4 tangent; // w - bitangent handedness
};

struct GPUMeshBuffers
{
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

struct GeometryPushConstants
{
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};

struct SceneCommonData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambient;
    glm::vec4 viewPos;
};

#define MAX_LIGHTS 64

struct LightsData
{
    // directional lights
    glm::vec4 dirDir[MAX_LIGHTS]; // w for power
    glm::vec4 dirColor[MAX_LIGHTS];

    // point lights
    glm::vec4 pointPos[MAX_LIGHTS];
    glm::vec4 pointColor[MAX_LIGHTS];
    glm::vec4 pointAtten[MAX_LIGHTS]; // x - linear, y - quad

    // spotlights
    glm::vec4 spotPos[MAX_LIGHTS];
    glm::vec4 spotDir[MAX_LIGHTS];
    glm::vec4 spotColor[MAX_LIGHTS];
    glm::vec4 spotAttenCutoff[MAX_LIGHTS]; // x - linear, y - quad, z - cutoff, w - outer cutoff
    
    int dirsUsed;
    int pointsUsed;
    int spotsUsed;
};

