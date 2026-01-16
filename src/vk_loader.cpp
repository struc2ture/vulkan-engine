#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

SceneImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath)
{
    SceneImage newImage {};
    int width, height, nrChannels;

    std::visit(
        fastgltf::visitor{
            [](auto& arg) {},
            [&](fastgltf::sources::URI &filePath) {
            assert(filePath.fileByteOffset == 0); // don't support offsets with stbi
            assert(filePath.uri.isLocalPath()); // can only load local files

            const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
            std::filesystem::path fullLocalPath = parentPath / std::filesystem::path{ path };
            unsigned char *data = stbi_load(fullLocalPath.generic_string().c_str(), &width, &height, &nrChannels, 4);
            if (data) {
                newImage.data = data;
                newImage.format = VK_FORMAT_R8G8B8A8_UNORM;
                newImage.width = width;
                newImage.height = height;
                newImage.path = fullLocalPath;
            }
        },
        [&](fastgltf::sources::Vector &vector) {
            unsigned char *data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()),
                &width, &height, &nrChannels, 4);
            if (data) {
                newImage.data = data;
                newImage.format = VK_FORMAT_R8G8B8A8_UNORM;
                newImage.width = width;
                newImage.height = height;
            }
        },
        [&](fastgltf::sources::BufferView &view) {
            auto& bufferView = asset.bufferViews[view.bufferViewIndex];
            auto& buffer = asset.buffers[bufferView.bufferIndex];

            std::visit(
                fastgltf::visitor {
                    // We only care about VectorWithMime here, because we
                    // specify LoadExternalBuffers, meaning all buffers
                    // are already loaded into a vector.
                    [](auto& arg) {},
                    [&](fastgltf::sources::Vector& vector) {
                    unsigned char *data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength),
                        &width, &height, &nrChannels, 4);
                    if (data) {
                        newImage.data = data;
                        newImage.format = VK_FORMAT_R8G8B8A8_UNORM;
                        newImage.width = width;
                        newImage.height = height;
                    }
                }
                },
                buffer.data
            );
        }
        },
        image.data
    );
    
    newImage.name = (newImage.data == nullptr) ? "<Texture not loaded>" : image.name.c_str();

    return newImage;
}

SceneImage load_image_data_from_file(std::filesystem::path path)
{
    SceneImage newImage {};
    int width, height, nrChannels;

    unsigned char *data = stbi_load(path.generic_string().c_str(), &width, &height, &nrChannels, 4);
    if (data) {
        newImage.data = data;
        newImage.format = VK_FORMAT_R8G8B8A8_UNORM;
        newImage.width = width;
        newImage.height = height;
    }
    
    newImage.name = (newImage.data == nullptr) ? "<Texture not loaded>" : path.filename().generic_string().c_str();

    return newImage;
}

std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath)
{
    AllocatedImage newImage {};

    int width, height, nrChannels;

    const bool mipped = true;

    std::visit(
        fastgltf::visitor{
            [](auto& arg) {},
            [&](fastgltf::sources::URI &filePath) {
                assert(filePath.fileByteOffset == 0); // don't support offsets with stbi
                assert(filePath.uri.isLocalPath()); // can only load local files

                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                std::filesystem::path fullLocalPath = parentPath / std::filesystem::path{ path };
                unsigned char *data = stbi_load(fullLocalPath.generic_string().c_str(), &width, &height, &nrChannels, 4);
                if (data) {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, mipped);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::Vector &vector) {
                unsigned char *data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()),
                    &width, &height, &nrChannels, 4);
                if (data) {
                    VkExtent3D imagesize;
                    imagesize.width = width;
                    imagesize.height = height;
                    imagesize.depth = 1;

                    newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, mipped);

                    stbi_image_free(data);
                }
            },
            [&](fastgltf::sources::BufferView &view) {
                auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                auto& buffer = asset.buffers[bufferView.bufferIndex];

                std::visit(
                    fastgltf::visitor {
                        // We only care about VectorWithMime here, because we
                        // specify LoadExternalBuffers, meaning all buffers
                        // are already loaded into a vector.
                        [](auto& arg) {},
                        [&](fastgltf::sources::Vector& vector) {
                            unsigned char *data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength),
                                &width, &height, &nrChannels, 4);
                            if (data) {
                                VkExtent3D imagesize;
                                imagesize.width = width;
                                imagesize.height = height;
                                imagesize.depth = 1;

                                newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, mipped);

                                stbi_image_free(data);
                            }
                        }
                    },
                    buffer.data
                );
            }
        },
        image.data
    );

    if (newImage.image == VK_NULL_HANDLE)
    {
        return {};
    }
    else
    {
        return newImage;
    }
}

std::optional<std::shared_ptr<Scene>> load_scene(VulkanEngine *engine, std::string_view filePath)
{
    fmt::println("Loading Scene: {}", filePath);

    std::filesystem::path path = filePath;

    auto scene = std::make_shared<Scene>(filePath.data(), path.filename().generic_string(), engine);

    fastgltf::Parser parser {};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load GLTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
        if (load)
        {
            gltf = std::move(load.get());
        }
        else
        {
            std::cerr << "Failed to load GLTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else
    {
        std::cerr << "Failed to determine GLTF container" << std::endl;
        return {};
    }

    for (fastgltf::Sampler &sampler : gltf.samplers)
    {
        auto newSampler = std::make_shared<SceneSampler>();
        newSampler->magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        newSampler->minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
        newSampler->mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
        newSampler->name = sampler.name.empty() ? (newSampler->magFilter == VK_FILTER_NEAREST ? "Nearest" : "Linear") : sampler.name.c_str();
        scene->samplers.push_back(newSampler);
    }

    for (fastgltf::Image &image : gltf.images)
    {
        auto newImage = std::make_shared<SceneImage>(load_image_data(gltf, image, path.parent_path()));
        scene->images.push_back(newImage);
    }

    for (fastgltf::Material &material : gltf.materials)
    {
        MaterialParameters material_params {};
        material_params.colorFactors.r = material.pbrData.baseColorFactor[0];
        material_params.colorFactors.g = material.pbrData.baseColorFactor[1];
        material_params.colorFactors.b = material.pbrData.baseColorFactor[2];
        material_params.colorFactors.a = material.pbrData.baseColorFactor[3];

        material_params.metal_rough_factors.r = material.pbrData.metallicFactor;
        material_params.metal_rough_factors.g = material.pbrData.roughnessFactor;

        auto newMaterial = std::make_shared<SceneMaterial>();
        newMaterial->name = material.name.c_str();
        newMaterial->hasColorImage = material.pbrData.baseColorTexture.has_value();
        if (newMaterial->hasColorImage)
        {
            size_t imageI = gltf.textures[material.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t samplerI = gltf.textures[material.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();
            newMaterial->colorImage = scene->images[imageI];
            newMaterial->colorSampler = scene->samplers[samplerI];
        }
        newMaterial->params = material_params;
        newMaterial->passType = MaterialPass::MainColor; // TODO: How to determine transparent pass

        scene->materials.push_back(newMaterial);
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        auto newMesh = std::make_shared<SceneMesh>();
        newMesh->name = mesh.name.c_str();

        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            ScenePrimitive newPrimitive {};
            newPrimitive.startIndex = (uint32_t)indices.size();
            newPrimitive.indexCount = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + (uint32_t)initial_vtx);
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 };
                        newvtx.color = glm::vec4 { 1.0f };
                        newvtx.uv_x = 0;
                        newvtx.uv_y = 0;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // load vertex normals
            {
                auto normals = p.findAttribute("NORMAL");
                if (normals != p.attributes.end())
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                        [&](glm::vec3 v, size_t index) {
                            vertices[initial_vtx + index].normal = v;
                        });
                }
            }

            // load UVs
            {
                auto uv = p.findAttribute("TEXCOORD_0");
                if (uv != p.attributes.end())
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                        [&](glm::vec2 v, size_t index) {
                            vertices[initial_vtx + index].uv_x = v.x;
                            vertices[initial_vtx + index].uv_y = v.y;
                        });
                }
            }

            // load vertex colors
            {
                auto colors = p.findAttribute("COLOR_0");
                if (colors != p.attributes.end())
                {
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                        [&](glm::vec4 v, size_t index) {
                            vertices[initial_vtx + index].color = v;
                        });
                }
            }

            if (p.materialIndex.has_value())
            {
                newPrimitive.material = scene->materials[p.materialIndex.value()];
            }
            else
            {
                newPrimitive.material = scene->materials[0];
            }
            newMesh->vertices = vertices;
            newMesh->indices = indices;

            newPrimitive.bounds = calculate_bounds(*newMesh, newPrimitive);

            newMesh->primitives.push_back(newPrimitive);
        }

        scene->meshes.push_back(newMesh);
    }

    uint64_t nodeId = 0;
    for (fastgltf::Node &node : gltf.nodes)
    {
        auto newNode = std::make_shared<SceneNode>();

        if (node.meshIndex.has_value())
        {
            newNode->Mesh = scene->meshes[node.meshIndex.value()];
        }
        newNode->NodeId = nodeId++;
        newNode->Name = node.name;

        std::visit(
            fastgltf::visitor{
                [&](fastgltf::Node::TransformMatrix matrix) {
                memcpy(&newNode->LocalTransform, matrix.data(), sizeof(matrix));
            },
            [&](fastgltf::Node::TRS transform) {
                glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                glm::mat4 tm = glm::translate(glm::mat4(1.0f), tl);
                glm::mat4 rm = glm::toMat4(rot);
                glm::mat4 sm = glm::scale(glm::mat4(1.0f), sc);

                newNode->LocalTransform = tm * rm * sm;
            }
            },
            node.transform
        );

        scene->nodes.push_back(newNode);
    }

    for (int i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node &node = gltf.nodes[i];
        auto sceneNode = scene->nodes[i];

        for (auto &childI : node.children)
        {
            sceneNode->Children.push_back(scene->nodes[childI]);
            scene->nodes[childI]->Parent = sceneNode;
        }
    }

    for (auto &node : scene->nodes)
    {
        if (node->Parent.lock() == nullptr)
        {
            scene->topNodes.push_back(node);
            node->RefreshTransform(glm::mat4{ 1.0f });
        }
    }

    scene->SyncToGPU();

    return scene;
}

Scene::Scene(std::string path_, std::string name_, VulkanEngine *engine_)
{
    path = path_;
    name = name_;
    engine = engine_;

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
    };
    descriptorPool.init(engine->_device, 10, sizes);
}

Scene::~Scene()
{
    _clearGPUData();
    descriptorPool.destroy_pools(engine->_device);
}

void Scene::SyncToGPU()
{
    for (auto &sampler : samplers)
    {
        VkSamplerCreateInfo samplerCreateInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
        samplerCreateInfo.minLod = 0;

        samplerCreateInfo.magFilter = sampler->magFilter;
        samplerCreateInfo.minFilter = sampler->minFilter;

        samplerCreateInfo.mipmapMode = sampler->mipmapMode;

        VkSampler vkSampler;
        vkCreateSampler(engine->_device, &samplerCreateInfo, nullptr, &vkSampler);

        sampler->vkSampler = vkSampler;
    }

    const bool mipmapped = false;
    for (auto &image : images)
    {
        if (image->data != nullptr)
        {
            VkExtent3D imageExtent {};
            imageExtent.width = image->width;
            imageExtent.height = image->height;
            imageExtent.depth = 1;

            image->allocatedImage = engine->create_image(image->data, imageExtent, image->format, VK_IMAGE_USAGE_SAMPLED_BIT, mipmapped);
        }
        else
        {
            image->allocatedImage = engine->_errorCheckerboardImage;
        }
    }

    materialDataBuffer = engine->create_buffer(sizeof(MaterialParameters) * materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto mappedParamsPtr = (MaterialParameters *)materialDataBuffer.info.pMappedData;

    int materialI = 0;

    for (auto &material : materials)
    {
        mappedParamsPtr[materialI] = material->params;

        StandardMaterial::Resources resources {};
        resources.ColorImage = material->hasColorImage ? material->colorImage->allocatedImage : engine->_whiteImage;
        resources.ColorSampler = material->hasColorImage ? material->colorSampler->vkSampler : engine->_defaultSamplerLinear;
        resources.MetalRoughImage = engine->_whiteImage;
        resources.MetalRoughSampler = engine->_defaultSamplerLinear;
        resources.MaterialParamDataBuffer = materialDataBuffer.buffer;
        resources.MaterialParamDataBufferOffset = materialI * sizeof(MaterialParameters);

        material->materialInstance = engine->MaterialBuilder.InstantiateMaterial(engine->_device, material->passType, resources, descriptorPool);

        materialI++;
    }

    for (auto &mesh : meshes)
    {
        mesh->meshBuffer = engine->uploadMesh(mesh->indices, mesh->vertices);
    }

    for (auto &node : topNodes)
    {
        node->RefreshTransform(glm::mat4{ 1.0f });
    }
}

void Scene::_clearGPUData()
{
    vkDeviceWaitIdle(engine->_device);

    engine->destroy_buffer(materialDataBuffer);

    for (auto &mesh : meshes)
    {
        engine->destroy_buffer(mesh->meshBuffer.indexBuffer);
        engine->destroy_buffer(mesh->meshBuffer.vertexBuffer);
    }

    for (auto &image : images)
    {
        if (image->allocatedImage.image == engine->_errorCheckerboardImage.image)
        {
            // don't destroy the default texture
            continue;
        }
        engine->destroy_image(image->allocatedImage);
    }

    for (auto &sampler : samplers)
    {
        vkDestroySampler(engine->_device, sampler->vkSampler, nullptr);
    }
}

void SceneNode::RefreshTransform(const glm::mat4 &parentMatrix)
{
    WorldTransform = parentMatrix * LocalTransform;
    for (auto &child : Children)
    {
        child->RefreshTransform(WorldTransform);
    }
}

void SceneNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    if (Mesh != nullptr)
    {
        glm::mat4 nodeMatrix = topMatrix * WorldTransform;

        for (auto &primitive : Mesh->primitives)
        {
            RenderObject def;
            def.indexCount = primitive.indexCount;
            def.firstIndex = primitive.startIndex;
            def.indexBuffer = Mesh->meshBuffer.indexBuffer.buffer;
            def.material = &primitive.material->materialInstance;
            def.bounds = primitive.bounds;
            def.transform = nodeMatrix;
            def.vertexBufferAddress = Mesh->meshBuffer.vertexBufferAddress;

            if (primitive.material->passType == MaterialPass::Transparent) {
                ctx.transparentSurfaces.push_back(def);
            } else {
                ctx.opaqueSurfaces.push_back(def);
            }
        }
    }

    for (auto &child : Children)
    {
        child->Draw(topMatrix, ctx);
    }
}

void Scene::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    for (auto &node : topNodes)
    {
        node->Draw(topMatrix, ctx);
    }
}

std::shared_ptr<SceneMesh> local_mesh_empty(std::string name)
{
    auto mesh = std::make_shared<SceneMesh>();
    mesh->name = name;
    return mesh;
}

std::shared_ptr<SceneMesh> local_mesh_cube(std::string name, std::shared_ptr<SceneMaterial> material)
{
    auto mesh = std::make_shared<SceneMesh>();
    mesh->name = name;

    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 1.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 1.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 1.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 1.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, -1.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, -1.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, -1.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, -1.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, 1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, 1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, -1.0), .uv_x = 0.0, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, -1.0), .uv_x = 1.0, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->indices = {
        0,1,2,    0,2,3,
        4,5,6,    4,6,7,
        8,9,10,   8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23
    };

    ScenePrimitive primitive {};
    primitive.startIndex = 0;
    primitive.indexCount = mesh->indices.size();
    primitive.bounds = calculate_bounds(*mesh, primitive);
    primitive.material = material;

    mesh->primitives.push_back(primitive);

    return mesh;
}

std::shared_ptr<SceneMesh> local_mesh_cylinder(std::string name, std::shared_ptr<SceneMaterial> material)
{
    auto mesh = std::make_shared<SceneMesh>();
    mesh->name = name;
    
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, -1.0, 0.0), .uv_x = 0.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(1.0, 1.0, 0.0), .uv_x = 0.0, .normal = glm::vec3(1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.924, -1.0, 0.383), .uv_x = 0.06, .normal = glm::vec3(0.924, 0.0, 0.383), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.924, 1.0, 0.383), .uv_x = 0.06, .normal = glm::vec3(0.924, 0.0, 0.383), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.707, -1.0, 0.707), .uv_x = 0.12, .normal = glm::vec3(0.707, 0.0, 0.707), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.707, 1.0, 0.707), .uv_x = 0.12, .normal = glm::vec3(0.707, 0.0, 0.707), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.383, -1.0, 0.924), .uv_x = 0.19, .normal = glm::vec3(0.383, 0.0, 0.924), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.383, 1.0, 0.924), .uv_x = 0.19, .normal = glm::vec3(0.383, 0.0, 0.924), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, -1.0, 1.0), .uv_x = 0.25, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, 1.0, 1.0), .uv_x = 0.25, .normal = glm::vec3(0.0, 0.0, 1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.383, -1.0, 0.924), .uv_x = 0.31, .normal = glm::vec3(-0.383, 0.0, 0.924), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.383, 1.0, 0.924), .uv_x = 0.31, .normal = glm::vec3(-0.383, 0.0, 0.924), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.707, -1.0, 0.707), .uv_x = 0.38, .normal = glm::vec3(-0.707, 0.0, 0.707), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.707, 1.0, 0.707), .uv_x = 0.38, .normal = glm::vec3(-0.707, 0.0, 0.707), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.924, -1.0, 0.383), .uv_x = 0.44, .normal = glm::vec3(-0.924, 0.0, 0.383), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.924, 1.0, 0.383), .uv_x = 0.44, .normal = glm::vec3(-0.924, 0.0, 0.383), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, -1.0, 0.0), .uv_x = 0.5, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-1.0, 1.0, 0.0), .uv_x = 0.5, .normal = glm::vec3(-1.0, 0.0, 0.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.924, -1.0, -0.383), .uv_x = 0.56, .normal = glm::vec3(-0.924, 0.0, -0.383), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.924, 1.0, -0.383), .uv_x = 0.56, .normal = glm::vec3(-0.924, 0.0, -0.383), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.707, -1.0, -0.707), .uv_x = 0.62, .normal = glm::vec3(-0.707, 0.0, -0.707), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.707, 1.0, -0.707), .uv_x = 0.62, .normal = glm::vec3(-0.707, 0.0, -0.707), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.383, -1.0, -0.924), .uv_x = 0.69, .normal = glm::vec3(-0.383, 0.0, -0.924), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(-0.383, 1.0, -0.924), .uv_x = 0.69, .normal = glm::vec3(-0.383, 0.0, -0.924), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, -1.0, -1.0), .uv_x = 0.75, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, 1.0, -1.0), .uv_x = 0.75, .normal = glm::vec3(0.0, 0.0, -1.0), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.383, -1.0, -0.924), .uv_x = 0.81, .normal = glm::vec3(0.383, 0.0, -0.924), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.383, 1.0, -0.924), .uv_x = 0.81, .normal = glm::vec3(0.383, 0.0, -0.924), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.707, -1.0, -0.707), .uv_x = 0.88, .normal = glm::vec3(0.707, 0.0, -0.707), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.707, 1.0, -0.707), .uv_x = 0.88, .normal = glm::vec3(0.707, 0.0, -0.707), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.924, -1.0, -0.383), .uv_x = 0.94, .normal = glm::vec3(0.924, 0.0, -0.383), .uv_y = 0.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.924, 1.0, -0.383), .uv_x = 0.94, .normal = glm::vec3(0.924, 0.0, -0.383), .uv_y = 1.0, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, 1.0, 0.0), .uv_x = 0.5, .normal = glm::vec3(0.0, 1.0, 0.0), .uv_y = 0.5, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->vertices.push_back(Vertex { .position = glm::vec3(0.0, -1.0, 0.0), .uv_x = 0.5, .normal = glm::vec3(0.0, -1.0, 0.0), .uv_y = 0.5, .color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)});
    mesh->indices = {
        // Sides (16 quads)
        0,1,3, 0,3,2,
        2,3,5, 2,5,4,
        4,5,7, 4,7,6,
        6,7,9, 6,9,8,
        8,9,11, 8,11,10,
        10,11,13, 10,13,12,
        12,13,15, 12,15,14,
        14,15,17, 14,17,16,
        16,17,19, 16,19,18,
        18,19,21, 18,21,20,
        20,21,23, 20,23,22,
        22,23,25, 22,25,24,
        24,25,27, 24,27,26,
        26,27,29, 26,29,28,
        28,29,31, 28,31,30,
        30,31,1, 30,1,0,

        // Top fan (center = 32)
        32,1,3,
        32,3,5,
        32,5,7,
        32,7,9,
        32,9,11,
        32,11,13,
        32,13,15,
        32,15,17,
        32,17,19,
        32,19,21,
        32,21,23,
        32,23,25,
        32,25,27,
        32,27,29,
        32,29,31,
        32,31,1,

        // Bottom fan (center = 33)
        33,2,0,
        33,4,2,
        33,6,4,
        33,8,6,
        33,10,8,
        33,12,10,
        33,14,12,
        33,16,14,
        33,18,16,
        33,20,18,
        33,22,20,
        33,24,22,
        33,26,24,
        33,28,26,
        33,30,28,
        33,0,30
    };

    ScenePrimitive primitive {};
    primitive.startIndex = 0;
    primitive.indexCount = mesh->indices.size();
    primitive.bounds = calculate_bounds(*mesh, primitive);
    primitive.material = material;

    mesh->primitives.push_back(primitive);

    return mesh;
}

std::shared_ptr<Scene> new_local_scene(VulkanEngine *engine, std::string name)
{
    auto scene = std::make_shared<Scene>(std::string{}, name.empty() ? "BoxScene" : name, engine);

    bool addDefault = name.empty();

    if (addDefault)
    {
        // material
        {
            auto newMaterial = std::make_shared<SceneMaterial>();
            newMaterial->name = "BoxMaterial";
            newMaterial->hasColorImage = false;

            MaterialParameters material_params {};
            material_params.colorFactors.r = 1.0f;
            material_params.colorFactors.g = 1.0f;
            material_params.colorFactors.b = 1.0f;
            material_params.colorFactors.a = 1.0f;

            material_params.metal_rough_factors.r = 0.0f;
            material_params.metal_rough_factors.g = 1.0f;

            newMaterial->params = material_params;

            newMaterial->passType = MaterialPass::MainColor;

            scene->materials.push_back(newMaterial);
        }

        // mesh
        {
            auto mesh = local_mesh_cube("BoxMesh", scene->materials[0]);
            scene->meshes.push_back(mesh);
        }

        // node
        {
            auto node = std::make_shared<SceneNode>();
            node->Name = "BoxNode";
            node->NodeId = 0;
            node->Mesh = scene->meshes[0];
            node->LocalTransform = glm::mat4 { 1.0f };
            node->RefreshTransform(glm::mat4{ 1.0f });

            scene->nodes.push_back(node);
            scene->topNodes.push_back(node);
        }

        scene->SyncToGPU();
    }

    return scene;
}

Bounds calculate_bounds(const SceneMesh &mesh, const ScenePrimitive &primitive)
{
    glm::vec3 minpos = mesh.vertices[mesh.indices[primitive.startIndex]].position;
    glm::vec3 maxpos = minpos;
    for (int i = primitive.startIndex; i < primitive.startIndex + primitive.indexCount; i++)
    {
        minpos = glm::min(minpos, mesh.vertices[mesh.indices[i]].position);
        maxpos = glm::max(maxpos, mesh.vertices[mesh.indices[i]].position);
    }

    Bounds bounds {};
    bounds.origin = (maxpos + minpos) / 2.0f;
    bounds.extents = (maxpos - minpos) / 2.0f;
    bounds.sphereRadius = glm::length(bounds.extents);

    return bounds;
}
