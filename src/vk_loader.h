#pragma once

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>
#include "vk_descriptors.h"

#include <fastgltf/glm_element_traits.hpp>

struct Bounds
{
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

// forward declaration
class VulkanEngine;

std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);

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
    MaterialParameters params;
    MaterialPass passType;

    MaterialInstance materialInstance;
};

struct ScenePrimitive
{
    uint32_t startIndex;
    uint32_t indexCount;
    Bounds bounds;
    std::shared_ptr<SceneMaterial> material;
};

struct SceneMesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ScenePrimitive> primitives;

    GPUMeshBuffers meshBuffer;
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

    void RefreshTransform(const glm::mat4 &parentMatrix);
    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);
};

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

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine *engine;

    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);

    Scene(std::string path, std::string name, VulkanEngine *engine);

    ~Scene();

    void SyncToGPU();

    void _clearGPUData();
};

std::shared_ptr<SceneMesh> local_mesh_empty(std::string name);
std::shared_ptr<SceneMesh> local_mesh_cube(std::string name, std::shared_ptr<SceneMaterial> material);
std::shared_ptr<SceneMesh> local_mesh_cylinder(std::string name, std::shared_ptr<SceneMaterial> material);

SceneImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);
SceneImage load_image_data_from_file(std::filesystem::path path);
void free_image_data(SceneImage &image); // TODO: Implement and put into Scene destructor
std::optional<std::shared_ptr<Scene>> load_scene(VulkanEngine *engine, std::string_view filePath);
std::shared_ptr<Scene> new_local_scene(VulkanEngine *engine, std::string name);
Bounds calculate_bounds(const SceneMesh &mesh, const ScenePrimitive &primitive);