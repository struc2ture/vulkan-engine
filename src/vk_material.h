#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"

enum class MaterialPass : uint8_t
{
    MainColor,
    Transparent,
    Other
};

struct MaterialParameters
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
