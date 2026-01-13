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

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine *engine, std::filesystem::path filePath)
{
    std::cout << "Loading GLTF: " << filePath << std::endl;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser {};

    auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
    if (load) {
        gltf = std::move(load.get());
    }
    else {
        fmt::print("Failed to load GLTF: {}\n", fastgltf::to_underlying(load.error()));
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        MeshAsset newmesh;

        newmesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

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

            newmesh.surfaces.push_back(newSurface);

            constexpr bool OverrideColors = false;
            if (OverrideColors) {
                for (Vertex &vtx : vertices) {
                    vtx.color = glm::vec4(vtx.normal, 1.0f);
                }
            }
        }

        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }

    return meshes;
}

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

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine *engine, std::string_view filePath)
{
    fmt::println("Loading GLTF: {}", filePath);

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF &file = *scene.get();

    fastgltf::Parser parser {};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

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

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
    };

    file.descriptorPool.init(engine->_device,  gltf.materials.size(), sizes);

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<GLTFImage>> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;
    std::vector<std::shared_ptr<GLTFSampler>> samplers;

    for (fastgltf::Sampler &sampler : gltf.samplers)
    {
        VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;

        sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler vkSampler;
        vkCreateSampler(engine->_device, &sampl, nullptr, &vkSampler);

        auto newSampler = std::make_shared<GLTFSampler>();
        newSampler->sampler = vkSampler;
        newSampler->name = string_VkFilter(sampl.magFilter);
        samplers.push_back(newSampler);
        file.samplers[newSampler->name] = newSampler;

        std::cout << "Loaded sampler: " << newSampler->name << std::endl;
    }

    for (fastgltf::Image &image : gltf.images)
    {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image, path.parent_path());

        std::shared_ptr<GLTFImage> newImage = std::make_shared<GLTFImage>();
        if (img.has_value())
        {
            newImage->name = image.name;
            newImage->image = img.value();
        }
        else
        {
            newImage->name = "<Texture not loaded>";
            newImage->image = engine->_errorCheckerboardImage;
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
        images.push_back(newImage);
        file.images[image.name.c_str()] = newImage;
    }

    file.materialDataBuffer = engine->create_buffer(sizeof(MaterialParameters) * gltf.materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    MaterialParameters *sceneMaterialParams = (MaterialParameters *)file.materialDataBuffer.info.pMappedData;

    for (fastgltf::Material &mat : gltf.materials)
    {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;
        newMat->name = mat.name;

        MaterialParameters params;
        params.colorFactors.r = mat.pbrData.baseColorFactor[0];
        params.colorFactors.g = mat.pbrData.baseColorFactor[1];
        params.colorFactors.b = mat.pbrData.baseColorFactor[2];
        params.colorFactors.a = mat.pbrData.baseColorFactor[3];

        params.metal_rough_factors.r = mat.pbrData.metallicFactor;
        params.metal_rough_factors.g = mat.pbrData.roughnessFactor;
        sceneMaterialParams[data_index] = params;

        MaterialPass passType = MaterialPass::MainColor;
        //if (mat.alphaMode == fastgltf::AlphaMode::Blend)
        //{
        //    passType = MaterialPass::Transparent;
        //}

        GLTFMetallic_Roughness::ResourceHeader materialResources;
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(MaterialParameters);
        
        int colorImageI = -1;
        int colorSamplerI = -1;
        if (mat.pbrData.baseColorTexture.has_value())
        {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img]->image;
            materialResources.colorSampler = samplers[sampler]->sampler;
            colorImageI = (int)img;
            colorSamplerI = (int)sampler;
        }

        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources, file.descriptorPool);
        newMat->params = params;

        // TODO: Forgot to check sentinel
        newMat->colorImage = images[colorImageI];
        newMat->colorSampler = samplers[colorSamplerI];

        data_index++;
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh : gltf.meshes)
    {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

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
                newSurface.material = materials[p.materialIndex.value()];
            }
            else
            {
                newSurface.material = materials[0];
            }

            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (int i = initial_vtx; i < vertices.size(); i++)
            {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }

            newSurface.bounds.origin = (maxpos + minpos) / 2.0f;
            newSurface.bounds.extents = (maxpos - minpos) / 2.0f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }

    uint64_t node_id = 0;
    for (fastgltf::Node &node : gltf.nodes)
    {
        std::shared_ptr<Node> newNode;

        if (node.meshIndex.has_value())
        {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else
        {
            newNode = std::make_shared<Node>();
        }
        newNode->node_id = node_id++;
        newNode->name = node.name;

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(
            fastgltf::visitor{
                [&](fastgltf::Node::TransformMatrix matrix) {
                    memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                },
                [&](fastgltf::Node::TRS transform) {
                    glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                    glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                    glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                    glm::mat4 tm = glm::translate(glm::mat4(1.0f), tl);
                    glm::mat4 rm = glm::toMat4(rot);
                    glm::mat4 sm = glm::scale(glm::mat4(1.0f), sc);

                    newNode->localTransform = tm * rm * sm;
                }
            },
            node.transform
        );
    }

    for (int i = 0; i < gltf.nodes.size(); i++)
    {
        fastgltf::Node &node = gltf.nodes[i];
        std::shared_ptr<Node> &sceneNode = nodes[i];

        for (auto &c : node.children)
        {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    for (auto &node : nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4 { 1.0f });
        }
    }

    return scene;
}

void LoadedGLTF::Draw(const glm::mat4 &topMatrix, DrawContext &ctx)
{
    for (auto &n : topNodes)
    {
        n->Draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll()
{
    VkDevice dv = creator->_device;

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);

    for (auto &[k, v] : meshes)
    {
        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto &[k, v] : images)
    {
        if (v->image.image == creator->_errorCheckerboardImage.image)
        {
            // don't destroy the default texture
            continue;
        }
        creator->destroy_image(v->image);
    }

    for (auto &[k, v] : samplers)
    {
        vkDestroySampler(dv, v->sampler, nullptr);
    }
}

LocalImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath)
{
    LocalImage newImage {};
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

std::optional<std::shared_ptr<LocalScene>> load_scene(VulkanEngine *engine, std::string_view filePath)
{
    fmt::println("Loading Scene: {}", filePath);

    std::filesystem::path path = filePath;

    auto scene = std::make_shared<LocalScene>();
    scene->path = filePath.data();
    scene->name = path.filename().generic_string();

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
        auto newSampler = std::make_shared<LocalSampler>();
        newSampler->name = sampler.name.c_str();
        newSampler->magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        newSampler->minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
        newSampler->mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
        scene->samplers.push_back(newSampler);
    }

    for (fastgltf::Image &image : gltf.images)
    {
        auto newImage = std::make_shared<LocalImage>(load_image_data(gltf, image, path.parent_path()));
        scene->images.push_back(newImage);
    }

    for (fastgltf::Material &mat : gltf.materials)
    {
        MaterialParameters material_params {};
        material_params.colorFactors.r = mat.pbrData.baseColorFactor[0];
        material_params.colorFactors.g = mat.pbrData.baseColorFactor[1];
        material_params.colorFactors.b = mat.pbrData.baseColorFactor[2];
        material_params.colorFactors.a = mat.pbrData.baseColorFactor[3];

        material_params.metal_rough_factors.r = mat.pbrData.metallicFactor;
        material_params.metal_rough_factors.g = mat.pbrData.roughnessFactor;

        auto newMaterial = std::make_shared<LocalMaterial>();
        newMaterial->name = mat.name.c_str();
        newMaterial->hasColorImage = mat.pbrData.baseColorTexture.has_value();
        if (newMaterial->hasColorImage)
        {
            size_t imageI = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t samplerI = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();
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
        auto newMesh = std::make_shared<LocalMesh>();
        newMesh->name = mesh.name.c_str();

        indices.clear();
        vertices.clear();

        for (auto &&p : mesh.primitives)
        {
            LocalPrimitive newPrimitive {};
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

            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (int i = initial_vtx; i < vertices.size(); i++)
            {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }

            newPrimitive.bounds.origin = (maxpos + minpos) / 2.0f;
            newPrimitive.bounds.extents = (maxpos - minpos) / 2.0f;
            newPrimitive.bounds.sphereRadius = glm::length(newPrimitive.bounds.extents);

            newMesh->primitives.push_back(newPrimitive);
        }

        newMesh->vertices = vertices;
        newMesh->indices = indices;

        scene->meshes.push_back(newMesh);
    }

    uint64_t node_id = 0;
    for (fastgltf::Node &node : gltf.nodes)
    {
        auto newNode = std::make_shared<LocalNode>();

        if (node.meshIndex.has_value())
        {
            newNode->loaded_mesh = scene->meshes[node.meshIndex.value()];
        }
        newNode->node_id = node_id++;
        newNode->name = node.name;

        std::visit(
            fastgltf::visitor{
                [&](fastgltf::Node::TransformMatrix matrix) {
                memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
            },
            [&](fastgltf::Node::TRS transform) {
                glm::vec3 tl(transform.translation[0], transform.translation[1], transform.translation[2]);
                glm::quat rot(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                glm::mat4 tm = glm::translate(glm::mat4(1.0f), tl);
                glm::mat4 rm = glm::toMat4(rot);
                glm::mat4 sm = glm::scale(glm::mat4(1.0f), sc);

                newNode->localTransform = tm * rm * sm;
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
            sceneNode->children.push_back(scene->nodes[childI]);
            scene->nodes[childI]->parent = sceneNode;
        }
    }

    for (auto &node : scene->nodes)
    {
        if (node->parent.lock() == nullptr)
        {
            scene->topNodes.push_back(node);
        }
    }

    return scene;
}

std::shared_ptr<LocalScene> new_local_scene(VulkanEngine *engine, std::string name)
{
    auto scene = std::make_shared<LocalScene>();
    scene->name = name;
    return scene;
}