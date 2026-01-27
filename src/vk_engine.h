#pragma once

#include <imgui.h>

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_scene.h"
#include "vk_material.h"
#include "camera.h"

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

struct BackgroundPushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct BackgroundEffect
{
	std::string name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	BackgroundPushConstants data;
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

struct RenderLightDirectional
{
	glm::vec3 direction;
	float power;
	glm::vec4 color;
};

struct RenderLightPoint
{
	glm::vec4 pos;
	glm::vec4 color;
	float attenuationLinear;
	float attenuationQuad;
};

struct RenderLightSpot
{
	glm::vec4 pos;
	glm::vec4 direction;
	glm::vec4 color;
	float attenuationLinear;
	float attenuationQuad;
	float cutoff;
	float outerCutoff;
};

struct RenderDebugIcon
{
	glm::vec3 position;
	glm::vec3 color;
	glm::vec2 size;
};

struct RenderDebugLine
{
	glm::vec3 start;
	glm::vec3 end;
	glm::vec3 startColor;
	glm::vec3 endColor;
	float thickness;
};

struct DrawContext
{
	std::vector<RenderObject> opaqueSurfaces;
	std::vector<RenderObject> transparentSurfaces;

	std::vector<RenderLightDirectional> directionalLights;
	std::vector<RenderLightPoint> pointLights;
	std::vector<RenderLightSpot> spotLights; 

	std::vector<RenderDebugIcon> debugIcons;
	std::vector<RenderDebugLine> debugLines;
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

struct ImguiPreviewTexture
{
	AllocatedImage image;
	ImTextureID texture_id;
};

class VulkanEngine
{
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

	VkPipeline _backgroundPipeline;
	VkPipelineLayout _backgroundPipelineLayout;

	VkPipeline _debugPipeline;
	VkPipelineLayout _debugPipelineLayout;
	VkDescriptorSetLayout _debugDescriptorSetLayout;

	GPUMeshBuffers _debugMeshBuffer;

	SceneImage _debugLightIcon;

	VkPipeline _debugLinePipeline;
	VkPipelineLayout _debugLinePipelineLayout;
	VkDescriptorSetLayout _debugLineDescriptorSetLayout;

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	std::vector<BackgroundEffect> backgroundEffects;
	size_t currentBackgroundEffect{0};

	bool resize_requested;

	SceneCommonData sceneData;

	LightsData lightsData;

	VkDescriptorSetLayout _sceneCommonDataDescriptorLayout;

	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _defaultNormalImage;
	AllocatedImage _errorCheckerboardImage;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	DrawContext _mainDrawContext;

	Camera mainCamera;

	EngineStats stats;

	bool _consoleMode;

	StandardMaterial StandardMaterial;
	RetroMaterial RetroMaterial;

	std::vector<std::shared_ptr<Scene>> _localScenes;
	std::weak_ptr<Scene> _inspectedScene;

	std::unordered_map<std::shared_ptr<SceneImage>, std::shared_ptr<ImguiPreviewTexture>> _imguiPreviewTextures;

	bool _imguiBackgroundWindow{ false };
	bool _imguiStatsWindow{ false };
	bool _imguiSceneListWindow{ true };
	bool _imguiDemoWindow{ false };
	bool _imguiRenderObjectsWindow{ false };
	bool _imguiSceneInspectorWindow{ true };
	bool _imguiCameraInspectorWindow{ false };

	void update_scene();

	void init();

	void cleanup();

	void draw();
	void draw_background(VkCommandBuffer cmd);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);
	void draw_debug_icons(VkCommandBuffer cmd);
	void draw_debug_lines(VkCommandBuffer cmd);

	void run();

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage &img);
	void destroy_buffer(const AllocatedBuffer &buffer);

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
	void init_debug_pipelines();
	void init_debug_line_pipelines();

	void init_default_data();

	void init_debug_objects();

	void init_imgui();

	void resize_swapchain();

	void imgui_uis();
	void imgui_background();
	void imgui_stats();
	void imgui_render_objects();
	void imgui_scene_list();
	void imgui_scene_inspector(std::shared_ptr<Scene> scene);
	void imgui_camera_inspector();

	void set_console_mode(bool state);
};

bool is_visible(const RenderObject &obj, const glm::mat4 &viewproj);