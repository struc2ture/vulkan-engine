#include "vk_scene.h"

#include "vk_types.h"
#include "vk_engine.h"
#include "vk_material.h"

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

    if (materials.size() > 0)
    {
        materialDataBuffer = engine->create_buffer(sizeof(StandardMaterialParameters) * materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        auto mappedParamsPtr = (StandardMaterialParameters *)materialDataBuffer.info.pMappedData;

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
            resources.MaterialParamDataBufferOffset = materialI * sizeof(StandardMaterialParameters);

            material->materialInstance = engine->StandardMaterial.InstantiateMaterial(engine->_device, material->passType, resources, descriptorPool);

            materialI++;
        }
    }
    else
    {
        materialDataBuffer = {};
    }

    if (retroMaterials.size() > 0)
    {
        retroMaterialDataBuffer = engine->create_buffer(sizeof(RetroMaterialParameters) * retroMaterials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        auto mappedParamsPtr = (RetroMaterialParameters *)retroMaterialDataBuffer.info.pMappedData;

        int materialI = 0;

        for (auto &material : retroMaterials)
        {
            mappedParamsPtr[materialI] = material->params;

            RetroMaterial::Resources resources {};

            resources.DiffuseImage = material->hasDiffuseImage ? material->diffuseImage->allocatedImage : engine->_whiteImage;
            resources.DiffuseSampler = material->hasDiffuseImage ? material->diffuseSampler->vkSampler : engine->_defaultSamplerLinear;
            
            resources.SpecularImage = material->hasSpecularImage ? material->specularImage->allocatedImage : engine->_whiteImage;
            resources.SpecularSampler = material->hasSpecularImage ? material->specularSampler->vkSampler : engine->_defaultSamplerLinear;

            resources.EmissionImage = material->hasEmissionImage ? material->emissionImage->allocatedImage : engine->_whiteImage;
            resources.EmissionSampler = material->hasEmissionImage ? material->emissionSampler->vkSampler : engine->_defaultSamplerLinear;
            
            resources.NormalImage = material->hasNormalImage ? material->normalImage->allocatedImage : engine->_defaultNormalImage;
            resources.NormalSampler = material->hasNormalImage ? material->normalSampler->vkSampler : engine->_defaultSamplerLinear;

            resources.ParallaxImage = material->hasParallaxImage ? material->parallaxImage->allocatedImage : engine->_blackImage;
            resources.ParallaxSampler = material->hasParallaxImage ? material->parallaxSampler->vkSampler : engine->_defaultSamplerLinear;

            resources.MaterialParamDataBuffer = retroMaterialDataBuffer.buffer;
            resources.MaterialParamDataBufferOffset = materialI * sizeof(RetroMaterialParameters);

            material->materialInstance = engine->RetroMaterial.InstantiateMaterial(engine->_device, material->passType, resources, descriptorPool);

            materialI++;
        }
    }
    else
    {
        retroMaterialDataBuffer = {};
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

    if (materialDataBuffer.buffer != VK_NULL_HANDLE) engine->destroy_buffer(materialDataBuffer);
    if (retroMaterialDataBuffer.buffer != VK_NULL_HANDLE) engine->destroy_buffer(retroMaterialDataBuffer);

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
            assert(primitive.material == nullptr);
            assert(primitive.retroMaterial != nullptr);

            RenderObject def;
            def.indexCount = primitive.indexCount;
            def.firstIndex = primitive.startIndex;
            def.indexBuffer = Mesh->meshBuffer.indexBuffer.buffer;
            def.material = &primitive.retroMaterial->materialInstance;
            def.bounds = primitive.bounds;
            def.transform = nodeMatrix;
            def.vertexBufferAddress = Mesh->meshBuffer.vertexBufferAddress;

            if (primitive.retroMaterial->passType == MaterialPass::Transparent) {
                ctx.transparentSurfaces.push_back(def);
            } else {
                ctx.opaqueSurfaces.push_back(def);
            }
        }
    }

    if (Light != nullptr)
    {
        glm::mat4 nodeMatrix = topMatrix * WorldTransform;

        glm::vec3 debugColor = glm::vec3(1.0f);
        switch (Light->Kind)
        {
        case SceneLight::Kind::Directional:
        {
            RenderLightDirectional directionalLight;
            directionalLight.direction = -nodeMatrix[2];
            directionalLight.power = Light->Power;
            directionalLight.color = Light->Color;
            ctx.directionalLights.push_back(directionalLight);

            debugColor = glm::vec3(0.0f, 1.0f, 0.0f);
        } break;
        case SceneLight::Kind::Point:
        {
            RenderLightPoint pointLight;
            pointLight.pos = nodeMatrix[3];
            pointLight.color = Light->Color;
            pointLight.attenuationLinear = Light->AttenuationLinear;
            pointLight.attenuationQuad = Light->AttenuationQuad;
            ctx.pointLights.push_back(pointLight);

            debugColor = glm::vec3(1.0f, 0.0f, 0.0f);
        } break;
        case SceneLight::Kind::Spotlight:
        {
            RenderLightSpot spotLight;
            spotLight.pos = nodeMatrix[3];
            spotLight.color = Light->Color;
            spotLight.direction = -nodeMatrix[2];
            spotLight.attenuationLinear = Light->AttenuationLinear;
            spotLight.attenuationQuad = Light->AttenuationQuad;
            spotLight.cutoff = glm::cos(glm::radians(Light->Cutoff));
            spotLight.outerCutoff = glm::cos(glm::radians(Light->OuterCutoff));
            ctx.spotLights.push_back(spotLight);

            debugColor = glm::vec3(0.0f, 0.0f, 1.0f);
        } break;
        }

        RenderDebugObject debugObject;
        debugObject.position = nodeMatrix[3];
        debugObject.color = debugColor;
        debugObject.size = glm::vec2(0.07f, 0.07f);
        ctx.debugObjects.push_back(debugObject);
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
