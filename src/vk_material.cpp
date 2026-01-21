#include "vk_material.h"

#include "vk_types.h"
#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include "vk_descriptors.h"

void StandardMaterial::BuildPipelines(VulkanEngine *engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/retro_material.frag.spv", engine->_device, &meshFragShader))
    {
        fmt::println("Error when building the mesh fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.vert.spv", engine->_device, &meshVertexShader))
    {
        fmt::println("Error when building the mesh vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GeometryPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    DescriptorLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_sceneCommonDataDescriptorLayout, DescriptorLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    OpaquePipeline.layout = newLayout;
    TransparentPipeline.layout = newLayout;

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    //pipelineBuilder.enable_blending_alphablend();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    pipelineBuilder._pipelineLayout = newLayout;

    OpaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // transparent pipeline
    //pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_blending_alphablend();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    TransparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

void StandardMaterial::DestroyPipelines(VkDevice device)
{
    vkDestroyPipelineLayout(device, TransparentPipeline.layout, nullptr); // Transparent and Opaque pipelines share the same layout
    vkDestroyPipeline(device, TransparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, OpaquePipeline.pipeline, nullptr);
    vkDestroyDescriptorSetLayout(device, DescriptorLayout, nullptr);
}

MaterialInstance StandardMaterial::InstantiateMaterial(VkDevice device, MaterialPass pass, const Resources &resources, DescriptorAllocatorGrowable &descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent)
    {
        matData.pipeline = &TransparentPipeline;
    }
    else
    {
        matData.pipeline = &OpaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, DescriptorLayout);

    Writer.clear();
    Writer.write_buffer(0, resources.MaterialParamDataBuffer, sizeof(StandardMaterialParameters), resources.MaterialParamDataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    Writer.write_image(1, resources.ColorImage.imageView, resources.ColorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.write_image(2, resources.MetalRoughImage.imageView, resources.MetalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.update_set(device, matData.materialSet);

    return matData;
}

void RetroMaterial::BuildPipelines(VulkanEngine *engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/retro_material.frag.spv", engine->_device, &meshFragShader))
    {
        fmt::println("Error when building the mesh fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.vert.spv", engine->_device, &meshVertexShader))
    {
        fmt::println("Error when building the mesh vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GeometryPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    //layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    //layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    DescriptorLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_sceneCommonDataDescriptorLayout, DescriptorLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    OpaquePipeline.layout = newLayout;
    TransparentPipeline.layout = newLayout;

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    //pipelineBuilder.enable_blending_alphablend();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    pipelineBuilder._pipelineLayout = newLayout;

    OpaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // transparent pipeline
    //pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_blending_alphablend();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    TransparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

void RetroMaterial::DestroyPipelines(VkDevice device)
{
    vkDestroyPipelineLayout(device, TransparentPipeline.layout, nullptr); // Transparent and Opaque pipelines share the same layout
    vkDestroyPipeline(device, TransparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, OpaquePipeline.pipeline, nullptr);
    vkDestroyDescriptorSetLayout(device, DescriptorLayout, nullptr);
}

MaterialInstance RetroMaterial::InstantiateMaterial(VkDevice device, MaterialPass pass, const Resources &resources, DescriptorAllocatorGrowable &descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent)
    {
        matData.pipeline = &TransparentPipeline;
    }
    else
    {
        matData.pipeline = &OpaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, DescriptorLayout);

    Writer.clear();
    Writer.write_buffer(0, resources.MaterialParamDataBuffer, sizeof(RetroMaterialParameters), resources.MaterialParamDataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    Writer.write_image(1, resources.DiffuseImage.imageView, resources.DiffuseSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    //Writer.write_image(2, resources.SpecularImage.imageView, resources.SpecularSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    //Writer.write_image(3, resources.NormalImage.imageView, resources.NormalSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.update_set(device, matData.materialSet);

    return matData;
}