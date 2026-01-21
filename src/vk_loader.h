#pragma once

#include <filesystem>
#include <unordered_map>

#include <fastgltf/glm_element_traits.hpp>

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_scene.h"

std::optional<AllocatedImage> load_image(VulkanEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);

std::shared_ptr<SceneMesh> local_mesh_empty(std::string name);
std::shared_ptr<SceneMesh> local_mesh_cube(std::string name, std::shared_ptr<SceneRetroMaterial> retroMaterial, std::shared_ptr<SceneMaterial> material = nullptr);
std::shared_ptr<SceneMesh> local_mesh_cylinder(std::string name, std::shared_ptr<SceneRetroMaterial> retroMaterial, std::shared_ptr<SceneMaterial> material = nullptr);

SceneImage load_image_data(fastgltf::Asset &asset, fastgltf::Image &image, std::filesystem::path parentPath);
SceneImage load_image_data_from_file(std::filesystem::path path);
void free_image_data(SceneImage &image); // TODO: Implement and put into Scene destructor
std::optional<std::shared_ptr<Scene>> load_scene(VulkanEngine *engine, std::string_view filePath);
std::shared_ptr<Scene> new_local_scene(VulkanEngine *engine, std::string name);
