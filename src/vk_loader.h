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

struct LocalImage
{
    std::string name;
    int width, height;
    VkFormat format;
    std::optional<std::filesystem::path> path;
    void *data;

    AllocatedImage allocatedImage;
};

struct LocalSampler
{
    std::string name;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;

    VkSampler vkSampler;
};

struct LocalMaterial
{
    std::string name;
    bool hasColorImage;
    std::shared_ptr<LocalImage> colorImage;
    std::shared_ptr<LocalSampler> colorSampler;
    MaterialParameters params;
    MaterialPass passType;

    MaterialInstance materialInstance;
};

struct LocalPrimitive
{
    uint32_t startIndex;
    uint32_t indexCount;
    Bounds bounds;
    std::shared_ptr<LocalMaterial> material;
};

struct LocalMesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<LocalPrimitive> primitives;

    GPUMeshBuffers meshBuffer;
};

struct LocalNode : public IRenderable
{
    std::string Name;
    uint64_t NodeId;

    std::weak_ptr<LocalNode> Parent;
    std::vector<std::shared_ptr<LocalNode>> Children;

    glm::mat4 LocalTransform;
    glm::mat4 WorldTransform;

    std::shared_ptr<LocalMesh> Mesh;

    void RefreshTransform(const glm::mat4 &parentMatrix);
    void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct LocalScene : public IRenderable
{
    std::string path;
    std::string name;

    std::vector<std::shared_ptr<LocalMesh>> meshes;
    std::vector<std::shared_ptr<LocalNode>> nodes;
    std::vector<std::shared_ptr<LocalImage>> images;
    std::vector<std::shared_ptr<LocalSampler>> samplers;
    std::vector<std::shared_ptr<LocalMaterial>> materials;
    std::vector<std::shared_ptr<LocalNode>> topNodes;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine *engine;

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

    ~LocalScene() { _clearGPUData(); };

    void SyncToGPU();

    void _clearGPUData();
};

std::shared_ptr<LocalMesh> local_mesh_empty(std::string name);
std::shared_ptr<LocalMesh> local_mesh_cube(std::string name, std::shared_ptr<LocalMaterial> material);
std::shared_ptr<LocalMesh> local_mesh_cylinder(std::string name, std::shared_ptr<LocalMaterial> material);

LocalImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);
LocalImage load_image_data_from_file(std::filesystem::path path);
void free_image_data(LocalImage &image); // TODO: Implement and put into LocalScene destructor
std::optional<std::shared_ptr<LocalScene>> load_scene(VulkanEngine *engine, std::string_view filePath);
std::shared_ptr<LocalScene> new_local_scene(VulkanEngine *engine, std::string name);
Bounds calculate_bounds(const LocalMesh &mesh, const LocalPrimitive &primitive);