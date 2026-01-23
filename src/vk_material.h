#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"

enum class MaterialPass : uint8_t
{
    MainColor,
    Transparent,
    Other
};

struct StandardMaterialParameters
{
    glm::vec4 colorFactors;
    glm::vec4 metal_rough_factors;
    glm::vec4 extra[14];
};

struct MaterialPipeline
{
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct MaterialInstance
{
    MaterialPipeline *pipeline;
    VkDescriptorSet materialSet;
    MaterialPass passType;
};

class VulkanEngine;

struct StandardMaterial
{
	struct Resources
	{
		AllocatedImage ColorImage;
		VkSampler ColorSampler;
		AllocatedImage MetalRoughImage;
		VkSampler MetalRoughSampler;
		VkBuffer MaterialParamDataBuffer;
		uint32_t MaterialParamDataBufferOffset;
	};

	MaterialPipeline OpaquePipeline;
	MaterialPipeline TransparentPipeline;

	VkDescriptorSetLayout DescriptorLayout;

	DescriptorWriter Writer;

	void BuildPipelines(VulkanEngine *engine);
	void DestroyPipelines(VkDevice device);

	MaterialInstance InstantiateMaterial(VkDevice device, MaterialPass pass, const Resources &resources, DescriptorAllocatorGrowable &descriptorAllocator);
};

struct RetroMaterialParameters
{
	glm::vec4 diffuse;
	glm::vec4 specular; // a - shininess
	glm::vec4 emission; // a - unused
	glm::vec4 extra[13];
};

struct RetroMaterial
{
	struct Resources
	{
		AllocatedImage DiffuseImage;
		VkSampler DiffuseSampler;
		AllocatedImage SpecularImage;
		VkSampler SpecularSampler;
		AllocatedImage EmissionImage;
		VkSampler EmissionSampler;
		AllocatedImage NormalImage;
		VkSampler NormalSampler;
		AllocatedImage ParallaxImage;
		VkSampler ParallaxSampler;
		VkBuffer MaterialParamDataBuffer;
		uint32_t MaterialParamDataBufferOffset;
	};

	MaterialPipeline OpaquePipeline;
	MaterialPipeline TransparentPipeline;

	VkDescriptorSetLayout DescriptorLayout;

	DescriptorWriter Writer;

	void BuildPipelines(VulkanEngine *engine);
	void DestroyPipelines(VkDevice device);

	MaterialInstance InstantiateMaterial(VkDevice device, MaterialPass pass, const Resources &resources, DescriptorAllocatorGrowable &descriptorAllocator);
};
