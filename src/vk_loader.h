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

struct LoadedImage
{
    std::string name;
    int width, height;
    VkFormat format;
    std::optional<std::filesystem::path> path;
    void *data;
};

struct LoadedSampler
{
    std::string name;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;
};

struct LoadedMaterial
{
    std::string name;
    bool hasColorImage;
    std::shared_ptr<LoadedImage> colorImage;
    std::shared_ptr<LoadedSampler> colorSampler;
    MaterialParameters params;
    MaterialPass passType;
};

struct LoadedPrimitive
{
    uint32_t startIndex;
    uint32_t indexCount;
    Bounds bounds;
    std::shared_ptr<LoadedMaterial> material;
};

struct LoadedMesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<LoadedPrimitive> primitives;
};

struct LoadedNode
{
    std::string name;
    uint64_t node_id;

    std::weak_ptr<LoadedNode> parent;
    std::vector<std::shared_ptr<LoadedNode>> children;

    glm::mat4 localTransform;

    std::shared_ptr<LoadedMesh> loaded_mesh;
};

struct LoadedScene
{
    std::string path;

    std::vector<std::shared_ptr<LoadedMesh>> meshes;
    std::vector<std::shared_ptr<LoadedNode>> nodes;
    std::vector<std::shared_ptr<LoadedImage>> images;
    std::vector<std::shared_ptr<LoadedSampler>> samplers;
    std::vector<std::shared_ptr<LoadedMaterial>> materials;
    std::vector<std::shared_ptr<LoadedNode>> topNodes;
};

LoadedImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);
void free_image_data(LoadedImage &image); // TODO: Implement and put into LoadedScene destructor
std::optional<std::shared_ptr<LoadedScene>> load_scene(VulkanEngine *engine, std::string_view filePath);
