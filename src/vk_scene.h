#pragma once

#include <filesystem>

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_material.h"

struct Bounds
{
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

struct SceneImage
{
    std::string name;
    int width, height;
    VkFormat format;
    std::optional<std::filesystem::path> path;
    void *data;

    AllocatedImage allocatedImage;
};

struct SceneSampler
{
    std::string name;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;

    VkSampler vkSampler;
};

struct SceneMaterial
{
    std::string name;
    bool hasColorImage;
    std::shared_ptr<SceneImage> colorImage;
    std::shared_ptr<SceneSampler> colorSampler;
    StandardMaterialParameters params;
    MaterialPass passType;

    MaterialInstance materialInstance;
};

struct SceneRetroMaterial
{
    std::string name;

    bool hasDiffuseImage;
    std::shared_ptr<SceneImage> diffuseImage;
    std::shared_ptr<SceneSampler> diffuseSampler;
    
    bool hasSpecularImage;
    std::shared_ptr<SceneImage> specularImage;
    std::shared_ptr<SceneSampler> specularSampler;

    bool hasEmissionImage;
    std::shared_ptr<SceneImage> emissionImage;
    std::shared_ptr<SceneSampler> emissionSampler;
    
    bool hasNormalImage;
    std::shared_ptr<SceneImage> normalImage;
    std::shared_ptr<SceneSampler> normalSampler;

    bool hasParallaxImage;
    std::shared_ptr<SceneImage> parallaxImage;
    std::shared_ptr<SceneSampler> parallaxSampler;
    
    RetroMaterialParameters params;
    MaterialPass passType;

    MaterialInstance materialInstance;
};

struct ScenePrimitive
{
    uint32_t startIndex;
    uint32_t indexCount;
    Bounds bounds;
    std::shared_ptr<SceneMaterial> material;
    std::shared_ptr<SceneRetroMaterial> retroMaterial;
};

struct SceneMesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ScenePrimitive> primitives;

    GPUMeshBuffers meshBuffer;
};

struct SceneLight
{
    enum class Kind
    {
        Directional,
        Point,
        Spotlight
    };

    Kind Kind;

    std::string Name;
    glm::vec4 Color;
    float Power; // [0.0, 1.0] - Directional
    float AttenuationLinear; // [0.0, ] - Point, Spotlight
    float AttenuationQuad; // [0.0, ] - Point, Spotlight
    float Cutoff; // degrees [0.0, 90.0] - Spotlight
    float OuterCutoff; // degrees [0.0, 90.0] - Spotlight
};

struct DrawContext;

struct SceneNode
{
    std::string Name;
    uint64_t NodeId;

    std::weak_ptr<SceneNode> Parent;
    std::vector<std::shared_ptr<SceneNode>> Children;

    glm::mat4 LocalTransform;
    glm::mat4 WorldTransform;

    // TODO: Sync from LocalTransform and back
    glm::vec3 DebugWindow_Position{0.0f};
    glm::vec3 DebugWindow_RotEuler{0.0f};
    glm::vec3 DebugWindow_Scale{1.0f};

    std::shared_ptr<SceneMesh> Mesh;
    std::shared_ptr<SceneLight> Light;

    void RefreshTransform(const glm::mat4 &parentMatrix);
    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);
};

class VulkanEngine;

struct Scene
{
    std::string path;
    std::string name;

    std::vector<std::shared_ptr<SceneMesh>> meshes;
    std::vector<std::shared_ptr<SceneNode>> nodes;
    std::vector<std::shared_ptr<SceneImage>> images;
    std::vector<std::shared_ptr<SceneSampler>> samplers;
    std::vector<std::shared_ptr<SceneMaterial>> materials;
    std::vector<std::shared_ptr<SceneNode>> topNodes;

    std::vector<std::shared_ptr<SceneRetroMaterial>> retroMaterials;

    std::vector<std::shared_ptr<SceneLight>> lights;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;
    AllocatedBuffer retroMaterialDataBuffer;

    VulkanEngine *engine;

    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);

    Scene(std::string path, std::string name, VulkanEngine *engine);

    ~Scene();

    void SyncToGPU();

    void _clearGPUData();
};

Bounds calculate_bounds(const SceneMesh &mesh, const ScenePrimitive &primitive);