// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

#include "vk_descriptors.h"
#include "vk_loader.h"

#include "camera.h"

#include <imgui.h>

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()> &&function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		// reverse iterate
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
		{
			(*it)();
		}

		deletors.clear();
	}
};

struct FrameData
{
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkSemaphore _swapchainSemaphore;
	VkFence _renderFence;

	DeletionQueue _deletionQueue;
	DescriptorAllocatorGrowable _frameDescriptors;
};

constexpr unsigned int FRAME_OVERLAP = 2;

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	const char *name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct RenderObject
{
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance *material;
	Bounds bounds;
	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
	std::vector<RenderObject> opaqueSurfaces;
	std::vector<RenderObject> transparentSurfaces;
};

struct EngineStats
{
	float frametime;
	int triangle_count;
	int drawcall_count;
	int render_object_count;
	float scene_update_time;
	float mesh_draw_time;
};

class VulkanEngine;

struct StandardMaterialResourceHeader
{
	AllocatedImage ColorImage;
	VkSampler ColorSampler;
	AllocatedImage MetalRoughImage;
	VkSampler MetalRoughSampler;
	VkBuffer MaterialParamDataBuffer;
	uint32_t MaterialParamDataBufferOffset;
};

struct StandardMaterialBuilder
{
	MaterialPipeline OpaquePipeline;
	MaterialPipeline TransparentPipeline;

	VkDescriptorSetLayout MaterialLayout;

	DescriptorWriter Writer;

	void BuildPipelines(VulkanEngine *engine);
	void ClearResources(VkDevice device);

	MaterialInstance WriteMaterial(VkDevice device, MaterialPass pass, const StandardMaterialResourceHeader &resources, DescriptorAllocatorGrowable &descriptorAllocator);
};

struct DrawImage
{
	std::string name;
	AllocatedImage image;
};

struct DrawSampler
{
	std::string name;
	VkSampler sampler;
};

struct DrawMaterial
{
	std::string name;
	std::shared_ptr<DrawImage> colorImage;
	std::shared_ptr<DrawSampler> colorSampler;
	MaterialParameters params;
	MaterialInstance data;
};

struct DrawPrimitive {
	uint32_t startIndex;
	uint32_t count;
	Bounds bounds;
	std::shared_ptr<DrawMaterial> material;
};

struct DrawMesh {
	std::string name;
	std::vector<DrawPrimitive> primitives;
	GPUMeshBuffers meshBuffers;
};

struct DrawNode : public IRenderable
{
	uint64_t NodeId;
	std::string Name;

	// parent pointer must be a weak pointer to avoid circular dependencies
	std::weak_ptr<DrawNode> Parent;
	std::vector<std::shared_ptr<DrawNode>> Children;

	glm::mat4 LocalTransform;
	glm::mat4 WorldTransform;

	std::shared_ptr<DrawMesh> Mesh;

	void RefreshTransform(const glm::mat4 &parentMatrix);

	void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct DrawScene : public IRenderable
{
	std::vector<std::shared_ptr<DrawMesh>> meshes;
    std::vector<std::shared_ptr<DrawNode>> nodes;
    std::vector<std::shared_ptr<DrawImage>> images;
    std::vector<std::shared_ptr<DrawMaterial>> materials;
    std::vector<std::shared_ptr<DrawSampler>> samplers;

	std::vector<std::shared_ptr<DrawNode>> topNodes;

	DescriptorAllocatorGrowable descriptorPool;

	AllocatedBuffer materialDataBuffer;

	VulkanEngine *creator;

	virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;

	~DrawScene() { clearGPUData(); };

	void clearGPUData();
};

struct ImguiPreviewTexture
{
	AllocatedImage image;
	ImTextureID texture_id;
};

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;
	
	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	VkExtent2D _drawExtent;
	float renderScale{1.0f};
	
	DescriptorAllocatorGrowable globalDescriptorAllocator;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{0};

	bool resize_requested;

	GPUSceneData sceneData;

	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	VkDescriptorSetLayout _singleImageDescriptorLayout;

	DrawContext mainDrawContext;

	Camera mainCamera;

	EngineStats stats;

	bool _consoleMode;

	StandardMaterialBuilder MaterialBuilder;

	std::vector<std::shared_ptr<LocalScene>> _localScenes;
	std::weak_ptr<LocalScene> _inspectedScene;

	std::unordered_map<std::shared_ptr<LocalImage>, std::shared_ptr<ImguiPreviewTexture>> _imguiPreviewTextures;

	bool _imguiBackgroundWindow{ false };
	bool _imguiStatsWindow{ false };
	bool _imguiSceneListWindow{ true };
	bool _imguiDemoWindow{ false };
	bool _imguiRenderObjectsWindow{ false };
	bool _imguiLocalSceneInspectorWindow{ true };
	bool _imguiCameraInspectorWindow{ false };

	void update_scene();

	static VulkanEngine& Get();

	void init();

	void cleanup();

	void draw();
	void draw_background(VkCommandBuffer cmd);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);

	void run();

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage &img);
	void destroy_buffer(const AllocatedBuffer &buffer);

	std::shared_ptr<DrawScene> upload_local_scene(std::shared_ptr<LocalScene> loaded_scene);

private:
	std::vector<VkSemaphore> _renderSemaphores;

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	void init_descriptors();

	void init_pipelines();
	void init_background_pipelines();

	void init_default_data();

	void init_imgui();

	void resize_swapchain();

	void imgui_uis();
	void imgui_background();
	void imgui_stats();
	void imgui_render_objects();
	void imgui_scene_list();
	void imgui_local_scene_inspector(std::shared_ptr<LocalScene> scene);
	void imgui_camera_inspector();

	void set_console_mode(bool state);
};

bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj);