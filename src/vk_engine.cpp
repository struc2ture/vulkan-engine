#include "vk_engine.h"

#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>
#include <glm/gtx/transform.hpp>

#include "vk_types.h"
#include "vk_initializers.h"
#include "vk_images.h"
#include "vk_pipelines.h"
#include "vk_loader.h"
#include "vk_scene.h"

constexpr bool USE_VALIDATION_LAYERS = true;

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_default_data();

    init_imgui();

    set_console_mode(true);

    _isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Example Vulkan Application")
        .request_validation_layers(USE_VALIDATION_LAYERS)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    // 1.3 features
    VkPhysicalDeviceVulkan13Features features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = true;
    features.synchronization2 = true;

    // 1.2 features
    VkPhysicalDeviceVulkan12Features features12 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    vkb::PhysicalDeviceSelector selector { vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags =  VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    // separate draw image
    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages {};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
       });
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo  commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo,  nullptr, &_frames[i]._commandPool));

        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device,  &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo  = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _immFence, nullptr); });

    for (size_t i = 0; i < _swapchainImages.size(); i++)
    {
        VkSemaphore renderSemaphore;
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &renderSemaphore));

        _renderSemaphores.push_back(renderSemaphore);
    }

    _mainDeletionQueue.push_function([=]()
        { 
            for (auto &semaphore : _renderSemaphores)
            {
                vkDestroySemaphore(_device, semaphore, nullptr);
            }
        }
    );
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (int i = 0; i < _swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

void VulkanEngine::init_descriptors()
{
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 }
    };

    globalDescriptorAllocator.init(_device, 10, sizes);

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _sceneCommonDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    {
        DescriptorWriter writer;
        writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(_device, _drawImageDescriptors );
    }

    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _sceneCommonDataDescriptorLayout, nullptr);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 }
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    MaterialBuilder.BuildPipelines(this);
}

void VulkanEngine::init_default_data()
{
    // default images
    {
        uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
        _whiteImage = create_image((void *)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

        uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
        _greyImage = create_image((void *)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

        uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
        _blackImage = create_image((void *)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

        // checkerboard image
        uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
        std::array<uint32_t, 16 * 16> pixels;
        for (int x = 0; x < 16; x++)
        {
            for (int y = 0; y < 16; y++)
            {
                pixels[y*16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
            }
        }
        _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    // default samplers
    {
        VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        sampl.magFilter = VK_FILTER_NEAREST;
        sampl.minFilter = VK_FILTER_NEAREST;
        vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

        sampl.magFilter = VK_FILTER_LINEAR;
        sampl.minFilter = VK_FILTER_LINEAR;
        vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);
    }

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
    });

    // Scene
    mainCamera.velocity = glm::vec3(0.0f);
    mainCamera.position = glm::vec3(0, 1, 3);
    //mainCamera.position = glm::vec3(30.f, -00.f, -085.f); // good for structure.glb scene
    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    auto scene = load_scene(this, "../../assets/struct_quinoa/struct_quinoa.gltf");
    assert(scene.has_value());
    _localScenes.push_back(scene.value());
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout {};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(BackgroundPushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_backgroundPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when loading the gradient_color.comp shader.\n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when loading the sky.comp shader.\n");
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.pNext = nullptr;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _backgroundPipelineLayout;
    computePipelineCreateInfo.stage = stageInfo;

    BackgroundEffect gradientEffect {};
    gradientEffect.layout = _backgroundPipelineLayout;
    gradientEffect.name = "Gradient";
    gradientEffect.data.data1 = glm::vec4(1, 0, 0, 1);
    gradientEffect.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradientEffect.pipeline));

    computePipelineCreateInfo.stage.module = skyShader;

    BackgroundEffect skyEffect {};
    skyEffect.layout = _backgroundPipelineLayout;
    skyEffect.name = "Sky";
    skyEffect.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);
    
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &skyEffect.pipeline));

    backgroundEffects.push_back(gradientEffect);
    backgroundEffects.push_back(skyEffect);

    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _backgroundPipelineLayout, nullptr);
        vkDestroyPipeline(_device, skyEffect.pipeline, nullptr);
        vkDestroyPipeline(_device, gradientEffect.pipeline, nullptr);
    });
}

void VulkanEngine::init_imgui()
{
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    ImGui::CreateContext();

    ImGui_ImplSDL2_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    // dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer,  buffer.allocation);
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);
    
    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo deviceAddressInfo { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAddressInfo);

    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void *data = staging.allocation->GetMappedData();

    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char *)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped)
    {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        if (mipmapped)
        {
            // will also transition to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            vkutil::generate_mipmaps(cmd, new_image.image, VkExtent2D{new_image.imageExtent.width, new_image.imageExtent.height});
        }
        else
        {
            vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    });

    destroy_buffer(uploadbuffer);

    return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage &img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {
        vkDeviceWaitIdle(_device);

        for (auto &[k, v] : _imguiPreviewTextures)
        {
            destroy_image(v->image);
        }

        _localScenes.clear();

        MaterialBuilder.DestroyPipelines(_device);


        for (int i = 0; i < FRAME_OVERLAP; i++)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool,  nullptr);

            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }

        _mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    update_scene();

    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr,&swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;
    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // draw to the draw image
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // copy draw image to swapchain image
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, _renderSemaphores[swapchainImageIndex]);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphores[swapchainImageIndex];
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }

    _frameNumber++;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    BackgroundEffect& effect = backgroundEffects[currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _backgroundPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _backgroundPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BackgroundPushConstants), &effect.data);

    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    stats.drawcall_count = 0;
    stats.triangle_count = 0;

    auto start = std::chrono::system_clock::now();

    // Scene common data
    VkDescriptorSet sceneCommonDataDescriptorSet;
    {
        // TODO: Why is this recreated every frame?
        AllocatedBuffer sceneCommonDataBuffer = create_buffer(sizeof(SceneCommonData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        get_current_frame()._deletionQueue.push_function([=, this]() {
            destroy_buffer(sceneCommonDataBuffer);
            });

        SceneCommonData *sceneUniformData = (SceneCommonData *)sceneCommonDataBuffer.allocation->GetMappedData();
        *sceneUniformData = sceneData;

        sceneCommonDataDescriptorSet = get_current_frame()._frameDescriptors.allocate(_device, _sceneCommonDataDescriptorLayout);

        DescriptorWriter writer;
        writer.write_buffer(0, sceneCommonDataBuffer.buffer, sizeof(SceneCommonData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writer.update_set(_device, sceneCommonDataDescriptorSet);

    }

    // sort opaque geometry by material and mesh to minimize pipeline state switches
    std::vector<uint32_t> opaque_draws;
    opaque_draws.reserve(mainDrawContext.opaqueSurfaces.size());

    for (size_t i = 0; i < mainDrawContext.opaqueSurfaces.size(); i++)
    {
        // frustum culling
        if (is_visible(mainDrawContext.opaqueSurfaces[i], sceneData.viewproj)) {
            opaque_draws.push_back(uint32_t(i));
        }
    }

    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto &iA, const auto &iB) {
        const RenderObject &A = mainDrawContext.opaqueSurfaces[iA];
        const RenderObject &B = mainDrawContext.opaqueSurfaces[iB];
        if (A.material == B.material)
        {
            return A.indexBuffer < B.indexBuffer;
        }
        else
        {
            return A.material < B.material;
        }
    });

    // begin rendering
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    // draw render objects
    MaterialPipeline *lastPipeline = nullptr;
    MaterialInstance *lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject &r)
    {
        if (r.material != lastMaterial)
        {
            if (r.material->pipeline != lastPipeline)
            {
                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1, &sceneCommonDataDescriptorSet, 0, nullptr);
                
                VkViewport viewport = {};
                viewport.x = 0;
                viewport.y = 0;
                viewport.width = _drawExtent.width;
                viewport.height = _drawExtent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;

                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor = {};
                scissor.offset.x = 0;
                scissor.offset.y = 0;
                scissor.extent.width = _drawExtent.width;
                scissor.extent.height = _drawExtent.height;

                vkCmdSetScissor(cmd, 0, 1, &scissor);
            }
            
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1, &r.material->materialSet, 0, nullptr);
        }

        if (r.indexBuffer != lastIndexBuffer)
        {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GeometryPushConstants pushConstants;
        pushConstants.worldMatrix = r.transform;
        pushConstants.vertexBuffer = r.vertexBufferAddress;
        vkCmdPushConstants(cmd, r.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GeometryPushConstants), &pushConstants);

        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);

        stats.drawcall_count ++;
        stats.triangle_count += r.indexCount / 3;
    };

    for (auto &i : opaque_draws)
    {
        draw(mainDrawContext.opaqueSurfaces[i]);
    }

    for (const RenderObject &r : mainDrawContext.transparentSurfaces)
    {
        draw(r);
    }

    stats.render_object_count = (int)opaque_draws.size() + (int)mainDrawContext.transparentSurfaces.size();

    vkCmdEndRendering(cmd);

    auto end = std::chrono::system_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.0f;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue

        auto start = std::chrono::system_clock::now();

        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }

            if (e.type == SDL_KEYDOWN && e.key.keysym.scancode == SDL_SCANCODE_GRAVE)
            {
                set_console_mode(!_consoleMode);
            }

            if (!_consoleMode)
            {
                mainCamera.processSDLEvent(e);
            }

            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested) {
            resize_swapchain();
        }

        //  imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (_consoleMode)
        {
            imgui_uis();
        }

        // make imgui calculate internal draw structures
        ImGui::Render();

        draw();

        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.0f;

    }
}

void StandardMaterial::BuildPipelines(VulkanEngine *engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", engine->_device, &meshFragShader))
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
    Writer.write_buffer(0, resources.MaterialParamDataBuffer, sizeof(MaterialParameters), resources.MaterialParamDataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    Writer.write_image(1, resources.ColorImage.imageView, resources.ColorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.write_image(2, resources.MetalRoughImage.imageView, resources.MetalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.update_set(device, matData.materialSet);

    return matData;
}

void VulkanEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    mainCamera.update();

    mainDrawContext.opaqueSurfaces.clear();
    mainDrawContext.transparentSurfaces.clear();
    
    for (auto &scene : _localScenes)
    {
        scene->Draw(glm::mat4{ 1.0f }, mainDrawContext);
    }

    sceneData.view = mainCamera.getViewMatrix();
    sceneData.proj = glm::perspective(glm::radians(70.0f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.0f, 0.1f);

    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    sceneData.ambientColor = glm::vec4(0.1f);
    sceneData.sunlightColor = glm::vec4(1.0f);
    sceneData.sunlightDirection = glm::vec4(0,1,0.5,1.0f);

    auto end = std::chrono::system_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.scene_update_time = elapsed.count() / 1000.f;
}

void VulkanEngine::imgui_uis()
{
    if (_imguiBackgroundWindow)
        imgui_background();

    if (_imguiStatsWindow)
        imgui_stats();

    if (_imguiRenderObjectsWindow)
        imgui_render_objects();

    if (_imguiSceneListWindow)
        imgui_scene_list();

    if (_imguiLocalSceneInspectorWindow)
        imgui_local_scene_inspector(_inspectedScene.lock());

    if (_imguiCameraInspectorWindow)
        imgui_camera_inspector();

    if (_imguiDemoWindow)
        ImGui::ShowDemoWindow(&_imguiDemoWindow);

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Windows"))
        {
            ImGui::MenuItem("Background", "", &_imguiBackgroundWindow);
            ImGui::MenuItem("Stats", "", &_imguiStatsWindow);
            ImGui::MenuItem("Render Objects", "", &_imguiRenderObjectsWindow);
            ImGui::MenuItem("Scene List", "", &_imguiSceneListWindow);
            ImGui::MenuItem("Local Scene Inspector", "", &_imguiLocalSceneInspectorWindow);
            ImGui::MenuItem("Camera Inspector", "", &_imguiCameraInspectorWindow);
            ImGui::MenuItem("Imgui Demo Window", "", &_imguiDemoWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void VulkanEngine::imgui_background()
{
    if (ImGui::Begin("Background", &_imguiBackgroundWindow))
    {
        ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.0f);

        BackgroundEffect& selectedEffect = backgroundEffects[currentBackgroundEffect];

        if (ImGui::BeginCombo("Effect##backgroundEffectCombo", selectedEffect.name.c_str()))
        {
            size_t effectI = 0;
            for (auto &effect : backgroundEffects)
            {
                const bool isSelected = currentBackgroundEffect == effectI;
                if (ImGui::Selectable(effect.name.c_str(), isSelected)) currentBackgroundEffect = effectI;
                if (isSelected) ImGui::SetItemDefaultFocus();
                effectI++;
            }
            ImGui::EndCombo();
        }

        ImGui::DragFloat4("data1", (float *) &selectedEffect.data.data1, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat4("data2", (float *) &selectedEffect.data.data2, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat4("data3", (float *) &selectedEffect.data.data3, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat4("data4", (float *) &selectedEffect.data.data4, 0.01f, 0.0f, 1.0f);
    }
    ImGui::End();
}

void VulkanEngine::imgui_stats()
{
    if (ImGui::Begin("Stats", &_imguiStatsWindow))
    {
        ImGui::Text("Frame time: %f ms", stats.frametime);
        ImGui::Text("Draw time: %f ms", stats.mesh_draw_time);
        ImGui::Text("Update time: %f ms", stats.scene_update_time);
        ImGui::Text("Triangles: %i", stats.triangle_count);
        ImGui::Text("Draws: %i", stats.drawcall_count);
        ImGui::Text("Render objects: %i", stats.render_object_count);
    }
    ImGui::End();
}

void VulkanEngine::imgui_render_objects()
{
    if (ImGui::Begin("Render Objects", &_imguiRenderObjectsWindow))
    {
        ImGui::Text("Opaque Surfaces: %zu", mainDrawContext.opaqueSurfaces.size());
        ImGui::Text("Transparent Surfaces: %zu", mainDrawContext.transparentSurfaces.size());
    }
    ImGui::End();
}

void VulkanEngine::imgui_scene_list()
{
    static char load_file_buffer[256] = {};
    static char new_scene_buffer[256] = {};

    if (ImGui::Begin("Scene List", &_imguiSceneListWindow))
    {
        for (auto it = _localScenes.begin(); it != _localScenes.end(); it++)
        {
            auto &scene = *it;
            ImGui::PushID(scene.get());
            ImGui::Text("%s", scene->name.c_str());
            ImGui::SameLine();
            if (ImGui::Button("Inspect"))
            {
                if (_inspectedScene.lock() == scene)
                {
                    _inspectedScene.reset();
                }
                else
                {
                    _inspectedScene = scene;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Delete"))
            {
                _localScenes.erase(it);
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }

        ImGui::InputText("##loadFileInput", load_file_buffer, 256);
        ImGui::SameLine();
        if (ImGui::Button("Load File"))
        {
            auto scene = load_scene(this, load_file_buffer);
            if (scene.has_value())
            {
                _localScenes.push_back(scene.value());
            }
            else
            {
                fmt::println("Failed to load file at {}", load_file_buffer);
            }
        }
        ImGui::InputText("##newSceneInput", new_scene_buffer, 256);
        ImGui::SameLine();
        if (ImGui::Button("Add new"))
        {
            _localScenes.push_back(new_local_scene(this, new_scene_buffer));
        }
    }
    ImGui::End();
}

void VulkanEngine::imgui_local_scene_inspector(std::shared_ptr<Scene> scene)
{
    if (ImGui::Begin("Local Scene Inspector", &_imguiLocalSceneInspectorWindow))
    {
        if (scene == nullptr)
        {
            ImGui::Text("No scene selected");
            ImGui::End();
            return;
        }

        if (!scene->name.empty())
        {
            ImGui::Text("%s", scene->name.c_str());
        }
        if (!scene->path.empty())
        {
            ImGui::Text("Path: %s", scene->path.c_str());
        }

        ImGui::Text("");

        bool sceneDirty = false;

        if (ImGui::TreeNode("Meshes", "Meshes: %d", scene->meshes.size()))
        {
            for (auto &mesh : scene->meshes)
            {
                ImGui::PushID(mesh.get());
                if (ImGui::TreeNode("", "%s", mesh->name.c_str()))
                {
                    int vertI = 0;
                    if (ImGui::TreeNode("Vertices", "Vertices (%d)", mesh->vertices.size()))
                    {
                        for (auto &vertex : mesh->vertices)
                        {
                            ImGui::PushID(vertI++);
                            if (ImGui::DragFloat3("", &vertex.position.x, 0.01f)) sceneDirty = true;
                            ImGui::PopID();
                        }
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNode("Indices", "Indices (%d)", mesh->indices.size()))
                    {
                        for (auto &index : mesh->indices)
                        {
                            ImGui::Text("%d", index);
                        }
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNode("Primitives", "Primitives (%d)", mesh->primitives.size()))
                    {
                        int primitiveI = 0;
                        for (auto &primitive : mesh->primitives)
                        {
                            ImGui::PushID(primitiveI);
                            if (ImGui::TreeNode("", "Primitive %d", primitiveI))
                            {
                                ImGui::Text("Start Index: %u", primitive.startIndex);
                                ImGui::Text("Index Count: %u", primitive.indexCount);
                                ImGui::Text("Bounds: %.1f, %.1f, %.1f", primitive.bounds.origin.x, primitive.bounds.origin.y, primitive.bounds.origin.z);

                                if (ImGui::BeginCombo("##materialCombo", primitive.material->name.c_str()))
                                {
                                    auto oldMaterial = primitive.material;
                                    for (auto &material : scene->materials)
                                    {
                                        const bool isSelected = material == primitive.material;
                                        if (ImGui::Selectable(material->name.c_str(), isSelected))
                                            primitive.material = material;

                                        if (isSelected)
                                            ImGui::SetItemDefaultFocus();
                                    }
                                    ImGui::EndCombo();
                                    if (oldMaterial != primitive.material) sceneDirty = true;
                                }

                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                            primitiveI++;
                        }
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            if (scene->materials.size() > 0)
            {
                const char* meshTypes[] = { "Empty", "Cube", "Cylinder" };
                static int meshType = 0;
                ImGui::Combo("##addMeshCombo", &meshType, meshTypes, 3);
                ImGui::SameLine();
                if (ImGui::Button("Add Mesh"))
                {
                    switch (meshType)
                    {
                    case 1:
                        scene->meshes.push_back(local_mesh_cube("Cube", scene->materials[0]));
                        break;
                    case 2:
                        scene->meshes.push_back(local_mesh_cylinder("Cylinder", scene->materials[0]));
                        break;
                    default:
                        scene->meshes.push_back(local_mesh_empty("Empty"));
                    }
                    sceneDirty = true;
                }
            }
            else
            {
                ImGui::BeginDisabled();
                ImGui::Button("Add Mesh");
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
                    ImGui::SetTooltip("No materials in the scene");
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Images", "Images: %d", scene->images.size()))
        {
            for (auto &image : scene->images)
            {
                ImGui::PushID(image.get());
                if (ImGui::TreeNode("", "%s", image->name.c_str()))
                {
                    ImGui::Text("%d x %d px", image->width, image->height);
                    std::shared_ptr<ImguiPreviewTexture> tex = nullptr;
                    if (_imguiPreviewTextures.find(image) != _imguiPreviewTextures.end())
                    {
                        tex = _imguiPreviewTextures[image];
                    }
                    else
                    {
                        tex = std::make_shared<ImguiPreviewTexture>();

                        VkExtent3D imageExtent {};
                        imageExtent.width = image->width;
                        imageExtent.height = image->height;
                        imageExtent.depth = 1;

                        tex->image = create_image(image->data, imageExtent, image->format, VK_IMAGE_USAGE_SAMPLED_BIT, false);
                        tex->texture_id = ImGui_ImplVulkan_AddTexture(_defaultSamplerLinear, tex->image.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                        _imguiPreviewTextures[image] = tex;
                    }
                    ImGui::Image(tex->texture_id, ImVec2(image->width, image->height));
                    if (image->path.has_value())
                    {
                        ImGui::Text("File at %s", image->path.value().generic_string().c_str());
                    }
                    else
                    {
                        ImGui::Text("File embedded");
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            static char addImageBuffer[256] = {};
            ImGui::InputText("##addImageInput", addImageBuffer, 256);
            ImGui::SameLine();
            if (ImGui::Button("Add Image"))
            {
                auto newImage = std::make_shared<SceneImage>(load_image_data_from_file(addImageBuffer));
                scene->images.push_back(newImage);
                sceneDirty = true;
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Samplers", "Samplers: %d", scene->samplers.size()))
        {
            for (auto &sampler : scene->samplers)
            {
                ImGui::PushID(sampler.get());
                if (ImGui::TreeNode("", "Sampler %s(%p)", sampler->name.c_str(), sampler.get()))
                {
                    ImGui::Text("Mag Filter: %s", string_VkFilter(sampler->magFilter));
                    ImGui::Text("Min Filter: %s", string_VkFilter(sampler->minFilter));
                    ImGui::Text("Mipmap Mode: %s", string_VkSamplerMipmapMode(sampler->mipmapMode));
                    
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            const char* samplerTypes[] = { "Nearest", "Linear"  };
            static int samplerType = 0;
            ImGui::Combo("##addSamplerCombo", &samplerType, samplerTypes, 2);
            ImGui::SameLine();
            if (ImGui::Button("Add Sampler"))
            {
                auto newSampler = std::make_shared<SceneSampler>();
                switch (samplerType)
                {
                case 0:
                    newSampler->magFilter = VK_FILTER_NEAREST;
                    newSampler->minFilter = VK_FILTER_NEAREST;
                    newSampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    newSampler->name = "Nearest";
                    break;
                default:
                    newSampler->magFilter = VK_FILTER_LINEAR;
                    newSampler->minFilter = VK_FILTER_LINEAR;
                    newSampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    newSampler->name = "Linear";
                }
                scene->samplers.push_back(newSampler);
                sceneDirty = true;
            }

            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Materials", "Materials: %d", scene->materials.size()))
        {
            for (auto &material : scene->materials)
            {
                ImGui::PushID(material.get());
                if (ImGui::TreeNode("", "%s", material->name.c_str()))
                {
                    if (material->hasColorImage)
                    {
                        ImGui::Text("Color image: %s", material->colorImage->name.c_str());
                        ImGui::Text("Color sampler: %p", material->colorSampler.get());
                        if (ImGui::Button("Remove Color Image"))
                        {
                            material->hasColorImage = false;
                            material->colorImage.reset();
                            material->colorSampler.reset();
                            sceneDirty = true;
                        }
                    }
                    else
                    {
                        //scene->images
                        if (scene->images.size() > 0 && scene->samplers.size() > 0)
                        {
                            static int selectedImage = 0;
                            if (ImGui::BeginCombo("##colorImageCombo", scene->images[selectedImage]->name.c_str()))
                            {
                                for (size_t imageI = 0; imageI < scene->images.size(); imageI++)
                                {
                                    const bool isSelected = imageI == selectedImage;
                                    if (ImGui::Selectable(scene->images[imageI]->name.c_str(), isSelected))
                                        selectedImage = imageI;

                                    if (isSelected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            static int selectedSampler = 0;
                            if (ImGui::BeginCombo("##colorSamplerCombo", scene->samplers[selectedSampler]->name.c_str()))
                            {
                                for (size_t samplerI = 0; samplerI < scene->samplers.size(); samplerI++)
                                {
                                    const bool isSelected = samplerI == selectedSampler;
                                    if (ImGui::Selectable(scene->samplers[samplerI]->name.c_str(), isSelected))
                                        selectedSampler = samplerI;

                                    if (isSelected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }

                            if (ImGui::Button("Add Color Image"))
                            {
                                material->colorImage = scene->images[selectedImage];
                                material->colorSampler = scene->samplers[selectedSampler];
                                material->hasColorImage = true;
                                sceneDirty = true;
                            }
                        }
                        else
                        {
                            ImGui::BeginDisabled();
                            ImGui::Button("Add Color Image");
                            ImGui::EndDisabled();
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
                                ImGui::SetTooltip("No image or sampler in the scene.");
                        }
                    }
                    if (ImGui::ColorEdit4("Color", &material->params.colorFactors.r)) sceneDirty = true;
                    if (ImGui::DragFloat("Metallic", &material->params.metal_rough_factors.r, 0.01f)) sceneDirty = true;
                    if (ImGui::DragFloat("Rougness", &material->params.metal_rough_factors.g, 0.01f)) sceneDirty = true;
                    ImGui::Text("Pass: %s", (material->passType == MaterialPass::MainColor) ? "Opaque" : "Transparent");
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            static char addMaterialBuffer[256];
            ImGui::InputText("##addMaterialInput", addMaterialBuffer, 256);
            ImGui::SameLine();
            if (ImGui::Button("Add Material"))
            {
                MaterialParameters material_params {};
                material_params.colorFactors.r = 1.0f;
                material_params.colorFactors.g = 1.0f;
                material_params.colorFactors.b = 1.0f;
                material_params.colorFactors.a = 1.0f;

                material_params.metal_rough_factors.r = 0.0f;
                material_params.metal_rough_factors.g = 1.0f;

                auto newMaterial = std::make_shared<SceneMaterial>();
                newMaterial->params = material_params;
                newMaterial->hasColorImage = false;
                newMaterial->passType = MaterialPass::MainColor;
                newMaterial->name = addMaterialBuffer;
                scene->materials.push_back(newMaterial);
                sceneDirty = true;
            }

            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Nodes", "Nodes: %d", scene->nodes.size()))
        {
            for (auto &node : scene->nodes)
            {
                ImGui::PushID(node.get());
                if (ImGui::TreeNode("", "%s[%llu]", node->Name.c_str(), node->NodeId))
                {
                    auto parent = node->Parent.lock();
                    if (parent != nullptr) ImGui::Text("Parent ID: %s[%llu]", parent->Name.c_str(), parent->NodeId);
                    else ImGui::Text("Parent ID: none");

                    if (ImGui::TreeNode("Local Tranfsorm"))
                    {
                        //ImGui::DragFloat4("##col0", &node->LocalTransform[0][0], 0.01f);
                        //ImGui::DragFloat4("##col1", &node->LocalTransform[1][0], 0.01f);
                        //ImGui::DragFloat4("##col2", &node->LocalTransform[2][0], 0.01f);
                        //ImGui::DragFloat4("##col3", &node->LocalTransform[3][0], 0.01f);
                        
                        if (ImGui::DragFloat3("Position", &node->DebugWindow_Position.x, 0.01f)) sceneDirty = true;
                        if (ImGui::DragFloat3("Rotation", &node->DebugWindow_RotEuler.x, 0.01f)) sceneDirty = true;
                        if (ImGui::DragFloat3("Scale", &node->DebugWindow_Scale.x, 0.01f)) sceneDirty = true;

                        // TODO: Sync from decomposed transform to LocalTransform and back as part of Node
                        node->LocalTransform = glm::mat4{ 1.0f };
                        node->LocalTransform = glm::translate(node->LocalTransform, node->DebugWindow_Position);
                        node->LocalTransform = glm::rotate(node->LocalTransform, node->DebugWindow_RotEuler.x, {1, 0, 0});
                        node->LocalTransform = glm::rotate(node->LocalTransform, node->DebugWindow_RotEuler.y, {0, 1, 0});
                        node->LocalTransform = glm::rotate(node->LocalTransform, node->DebugWindow_RotEuler.z, {0, 0, 1});
                        node->LocalTransform = glm::scale(node->LocalTransform, node->DebugWindow_Scale);

                        ImGui::TreePop();
                    }

                    if (ImGui::TreeNode("Children", "Children: %d", node->Children.size()))
                    {
                        for (auto &child : node->Children)
                        {
                            ImGui::Text("%s[%llu]", child->Name.c_str(), child->NodeId);
                        }
                        ImGui::TreePop();
                    }
                    
                    if (node->Mesh != nullptr) ImGui::Text("Mesh: %s", node->Mesh->name.c_str());
                    else ImGui::Text("Mesh: none");

                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            if (scene->meshes.size() > 0)
            {
                static int selectedMesh = 0;
                if (ImGui::BeginCombo("##addNodeMeshCombo", scene->meshes[selectedMesh]->name.c_str()))
                {
                    for (size_t meshI = 0; meshI < scene->meshes.size(); meshI++)
                    {
                        const bool isSelected = meshI == selectedMesh;
                        if (ImGui::Selectable(scene->meshes[meshI]->name.c_str(), isSelected))
                            selectedMesh = meshI;

                        if (isSelected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                static char addNodeBuffer[256] = {};
                ImGui::InputText("##addNodeInput", addNodeBuffer, 256);
                ImGui::SameLine();

                if (ImGui::Button("Add Node"))
                {
                    auto node = std::make_shared<SceneNode>();
                    node->Name = addNodeBuffer;
                    node->NodeId = (uint64_t)scene->meshes.size();
                    node->Mesh = scene->meshes[selectedMesh];
                    node->LocalTransform = glm::mat4 { 1.0f };

                    scene->nodes.push_back(node);
                    scene->topNodes.push_back(node);

                    sceneDirty = true;
                }
            }
            else
            {
                ImGui::BeginDisabled();
                ImGui::Button("Add Node");
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
                    ImGui::SetTooltip("No meshes in the scene");
            }

            ImGui::TreePop();
        }

        if (sceneDirty)
        {
            scene->_clearGPUData();
            scene->SyncToGPU();
        }
    }
    ImGui::End();
}

void VulkanEngine::imgui_camera_inspector()
{
    if (ImGui::Begin("Camera Inspector", &_imguiCameraInspectorWindow))
    {
        ImGui::InputFloat3("Position", &mainCamera.position.x);
    }
    ImGui::End();
}

void VulkanEngine::set_console_mode(bool state)
{
    _consoleMode = state;
    SDL_SetRelativeMouseMode((SDL_bool) !_consoleMode);
}

bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj)
{
    std::array<glm::vec3, 8> corners
    {
        glm::vec3 { 1, 1, 1},
        glm::vec3 { 1, 1, -1},
        glm::vec3 { 1, -1, 1},
        glm::vec3 { 1, -1, -1},
        glm::vec3 { -1, 1, 1},
        glm::vec3 { -1, 1, -1},
        glm::vec3 { -1, -1, 1},
        glm::vec3 { -1, -1, -1},
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min = { 1.5, 1.5, 1.5};
    glm::vec3 max = { -1.5, -1.5, -1.5};

    for (int c = 0; c < 8; c++)
    {
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.0f);

        // perspective correction (after multiplying by viewproj)
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3 { v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3 { v.x, v.y, v.z }, max);
    }

    if (min.z > 1.0f || max.z < 0.0f || min.x > 1.0f || max.x < -1.0f || min.y > 1.0f || max.y < -1.0f)
    {
        return false;
    }
    else
    {
        return true;
    }
}