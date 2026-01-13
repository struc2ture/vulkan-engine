#pragma once

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>
#include "vk_descriptors.h"

#include <fastgltf/glm_element_traits.hpp>

struct GLTFImage
{
    std::string name;
    AllocatedImage image;
};

struct GLTFSampler
{
    std::string name;
    VkSampler sampler;
};

struct GLTFMaterial
{
    std::string name;
    std::shared_ptr<GLTFImage> colorImage;
    std::shared_ptr<GLTFSampler> colorSampler;
    MaterialParameters params;
    MaterialInstance data;
};

struct Bounds
{
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

// forward declaration
class VulkanEngine;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine *engine, std::filesystem::path filePath);

struct LoadedGLTF : public IRenderable
{
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, std::shared_ptr<GLTFImage>> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;
    std::unordered_map<std::string, std::shared_ptr<GLTFSampler>> samplers;

    std::vector<std::shared_ptr<Node>> topNodes;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine *creator;

    ~LoadedGLTF() { clearAll(); };

    virtual void Draw(const glm::mat4& topMatrix, DrawContext &ctx);

private:
    void clearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine *engine, std::string_view filePath);

std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);

struct LocalImage
{
    std::string name;
    int width, height;
    VkFormat format;
    std::optional<std::filesystem::path> path;
    void *data;
};

struct LocalSampler
{
    std::string name;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;
};

struct LocalMaterial
{
    std::string name;
    bool hasColorImage;
    std::shared_ptr<LocalImage> colorImage;
    std::shared_ptr<LocalSampler> colorSampler;
    MaterialParameters params;
    MaterialPass passType;
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
};

struct LocalNode
{
    std::string name;
    uint64_t node_id;

    std::weak_ptr<LocalNode> parent;
    std::vector<std::shared_ptr<LocalNode>> children;

    glm::mat4 localTransform;

    std::shared_ptr<LocalMesh> loaded_mesh;
};

struct LocalScene
{
    std::string path;
    std::string name;

    std::vector<std::shared_ptr<LocalMesh>> meshes;
    std::vector<std::shared_ptr<LocalNode>> nodes;
    std::vector<std::shared_ptr<LocalImage>> images;
    std::vector<std::shared_ptr<LocalSampler>> samplers;
    std::vector<std::shared_ptr<LocalMaterial>> materials;
    std::vector<std::shared_ptr<LocalNode>> topNodes;
};

LocalImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);
void free_image_data(LocalImage &image); // TODO: Implement and put into LocalScene destructor
std::optional<std::shared_ptr<LocalScene>> load_scene(VulkanEngine *engine, std::string_view filePath);
std::shared_ptr<LocalScene> new_local_scene(VulkanEngine *engine, std::string name);
Bounds calculate_bounds(const LocalMesh &mesh, const LocalPrimitive &primitive);