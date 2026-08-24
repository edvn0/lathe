#include <csignal>
#include <memory>
#include <random>
#include <ranges>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <future>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <entt/entt.hpp>

#include "allocator.hxx"
#include "application.hxx"
#include "components.hxx"
#include "config.hxx"
#include "context.hxx"
#include "debug_renderer.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "imgui_renderer.hxx"
#include "implot.h"
#include "logger.hxx"
#include "physics.hxx"
#include "physics_world.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"
#include "renderdoc.hxx"
#include "renderer.hxx"
#include "renderer_application_policy.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "swapchain.hxx"

namespace {

    constexpr auto draw_point_light = [](Components::PointLight &point_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &point_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &point_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &point_light.range, 0.5F, 100.0F);
        return changed;
    };

    constexpr auto draw_spot_light = [](Components::SpotLight &spot_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &spot_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &spot_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &spot_light.range, 0.5F, 100.0F);
        changed |= ImGui::SliderFloat("Inner cone", &spot_light.inner_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        changed |= ImGui::SliderFloat("Outer cone", &spot_light.outer_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        return changed;
    };

    constexpr auto draw_rows = [](auto &index, entt::registry &registry, auto &&view, auto &&draw_light) {
        for (auto [entity, transform, light, meta]: view.each()) {
            ImGui::PushID(static_cast<int>(index++));
            if (ImGui::TreeNode(meta.name.c_str())) {
                bool changed = ImGui::DragFloat3("Position", &transform.position.x, 0.1F);
                changed |= draw_light(light);

                if (changed) {
                    using LightT = std::decay_t<decltype(light)>;
                    registry.patch<LightT>(entity);
                }

                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    };

    template<typename T>
    concept HasDepth = requires(T t) {
        { t.depth };
    };

    template<typename T>
    concept HasHeight = requires(T t) {
        { t.height };
    };

    template<typename T>
    concept HasWidth = requires(T t) {
        { t.width };
    };

    constexpr auto compare = []<typename A, typename B>(const A &a, const B &b) -> bool {
        if constexpr (HasDepth<A> && HasDepth<B>) {
            return a.width == b.width && a.height == b.height && a.depth == b.depth;
        } else if constexpr ((HasHeight<A> && !HasDepth<A>) && (HasHeight<B> && !HasDepth<B>) ) {
            return a.width == b.width && a.height == b.height;
        } else if constexpr ((HasWidth<A> && !HasHeight<A>) && (HasWidth<B> && !HasHeight<B>) ) {
            return a.width == b.width;
        } else {
            return false;
        }
    };


#ifndef NDEBUG
    constexpr bool enable_validation = true;

    constexpr std::array validation_layers{
            "VK_LAYER_KHRONOS_validation",
    };
#else
    constexpr bool enable_validation = false;

    constexpr std::array<const char *, 0> validation_layers{};
#endif


    constexpr auto widget = [](const std::string_view name, auto &&f) -> bool {
        if (!ImGui::Begin(name.data())) {
            ImGui::End();
            return false;
        }

        f();

        ImGui::End();
        return false;
    };


    auto vk_result_name(VkResult result) noexcept -> std::string_view {
        switch (result) {
            case VK_SUCCESS:
                return "VK_SUCCESS";

            case VK_NOT_READY:
                return "VK_NOT_READY";

            case VK_TIMEOUT:
                return "VK_TIMEOUT";

            case VK_EVENT_SET:
                return "VK_EVENT_SET";

            case VK_EVENT_RESET:
                return "VK_EVENT_RESET";

            case VK_INCOMPLETE:
                return "VK_INCOMPLETE";

            case VK_ERROR_OUT_OF_HOST_MEMORY:
                return "VK_ERROR_OUT_OF_HOST_MEMORY";

            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                return "VK_ERROR_OUT_OF_DEVICE_MEMORY";

            case VK_ERROR_INITIALIZATION_FAILED:
                return "VK_ERROR_INITIALIZATION_FAILED";

            case VK_ERROR_DEVICE_LOST:
                return "VK_ERROR_DEVICE_LOST";

            case VK_ERROR_MEMORY_MAP_FAILED:
                return "VK_ERROR_MEMORY_MAP_FAILED";

            case VK_ERROR_LAYER_NOT_PRESENT:
                return "VK_ERROR_LAYER_NOT_PRESENT";

            case VK_ERROR_EXTENSION_NOT_PRESENT:
                return "VK_ERROR_EXTENSION_NOT_PRESENT";

            case VK_ERROR_FEATURE_NOT_PRESENT:
                return "VK_ERROR_FEATURE_NOT_PRESENT";

            case VK_ERROR_INCOMPATIBLE_DRIVER:
                return "VK_ERROR_INCOMPATIBLE_DRIVER";

            case VK_ERROR_TOO_MANY_OBJECTS:
                return "VK_ERROR_TOO_MANY_OBJECTS";

            case VK_ERROR_FORMAT_NOT_SUPPORTED:
                return "VK_ERROR_FORMAT_NOT_SUPPORTED";

            case VK_ERROR_FRAGMENTED_POOL:
                return "VK_ERROR_FRAGMENTED_POOL";

            case VK_ERROR_SURFACE_LOST_KHR:
                return "VK_ERROR_SURFACE_LOST_KHR";

            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
                return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";

            default:
                return "VK_UNKNOWN_RESULT";
        }
    }

    auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void {
        error("{} failed: {} ({})", operation, vk_result_name(result), static_cast<int>(result));
    }

    auto validation_layer_available() noexcept -> bool {
        std::uint32_t layer_count = 0;

        auto result = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateInstanceLayerProperties(count)", result);

            return false;
        }

        std::vector<VkLayerProperties> layers(layer_count);

        result = vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateInstanceLayerProperties(list)", result);

            return false;
        }

        return std::ranges::any_of(layers, [](VkLayerProperties const &layer) {
            return std::string_view{layer.layerName} == validation_layers.front();
        });
    }

    VKAPI_ATTR auto VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     VkDebugUtilsMessengerCallbackDataEXT const *callback_data,
                                                     void *) noexcept -> VkBool32 {
        auto const *message = callback_data != nullptr && callback_data->pMessage != nullptr
                                      ? callback_data->pMessage
                                      : "<no validation message>";

        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
            error("Vulkan validation: {}", message);
        } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
            warn("Vulkan validation: {}", message);
        } else {
            debug("Vulkan validation: {}", message);
        }

        return VK_FALSE;
    }

    auto make_debug_messenger_create_info() noexcept -> VkDebugUtilsMessengerCreateInfoEXT {
        return VkDebugUtilsMessengerCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .pNext = nullptr,
                .flags = 0,
                .messageSeverity =
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = vulkan_debug_callback,
                .pUserData = nullptr,
        };
    }

    auto glfw_error_callback(int error_code, char const *description) noexcept -> void {
        error("GLFW error {}: {}", error_code, description != nullptr ? description : "<no description>");
    }

    enum class ScreenType {
        windowed,
        fullscreen,
        borderless,
    };

    auto parse_screen_type(int argc, char **argv) noexcept -> ScreenType {
        constexpr std::string_view prefix = "--screen-type=";

        for (auto i = 1; i < argc; ++i) {
            std::string_view const arg = argv[i];

            if (!arg.starts_with(prefix)) {
                continue;
            }

            auto const value = arg.substr(prefix.size());

            if (value == "windowed") {
                return ScreenType::windowed;
            }
            if (value == "fullscreen") {
                return ScreenType::fullscreen;
            }
            if (value == "borderless") {
                return ScreenType::borderless;
            }

            warn("Unknown --screen-type value '{}'; falling back to fullscreen", value);
        }

        return ScreenType::fullscreen;
    }

    auto initialize_glfw(VulkanContext &context, ScreenType screen_type) noexcept -> bool {
        glfwSetErrorCallback(glfw_error_callback);

        auto renderdoc = renderdoc_init();

        info("Was created via renderdoc: {}", renderdoc.is_active());

#if defined(__linux__)
        // RenderDoc does not support capturing Wayland surfaces, so force X11 when active.
        glfwInitHint(GLFW_PLATFORM, renderdoc.is_active() ? GLFW_PLATFORM_X11 : GLFW_ANY_PLATFORM);
        warn("Chosing: {} as platform.", renderdoc.is_active() ? "X11" : "auto-detected");
#endif

        if (glfwInit() != GLFW_TRUE) {
            error("glfwInit failed");
            return false;
        }

        glfwSetMonitorCallback([](GLFWmonitor *monitor, int event) {
            auto const *name = glfwGetMonitorName(monitor);

            if (event == GLFW_CONNECTED) {
                info("Monitor connected: {}", name != nullptr ? name : "<unknown>");
            } else if (event == GLFW_DISCONNECTED) {
                info("Monitor disconnected: {}", name != nullptr ? name : "<unknown>");
            }
        });

        if (glfwVulkanSupported() != GLFW_TRUE) {
            error("GLFW could not find a Vulkan loader "
                  "or required platform support");

            return false;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_DECORATED, screen_type == ScreenType::borderless ? GLFW_FALSE : GLFW_TRUE);

        std::int32_t monitor_count{};
        glfwGetMonitors(&monitor_count);
        auto monitors = glfwGetMonitors(&monitor_count);
        std::span<GLFWmonitor *> ms{monitors, static_cast<std::uint32_t>(monitor_count)};
        GLFWmonitor *selected = nullptr;
        std::int32_t max_width = std::numeric_limits<std::int32_t>::min();
        debug("Evaluating monitors...");
        for (auto &m: ms) {
            auto name = glfwGetMonitorName(m);
            auto mode = glfwGetVideoMode(m);
            debug("\t{} - {}", name, mode->width);
            if (mode->width > max_width) {
                selected = m;
                max_width = mode->width;
            }
        }
        debug("Done.");

        if (selected == nullptr) {
            error("Could not find a monitor");
            return false;
        }


        auto monitor = selected;
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

        switch (screen_type) {
            case ScreenType::fullscreen:
                context.window = glfwCreateWindow(mode->width, mode->height, "VK", monitor, NULL);
                break;

            case ScreenType::borderless:
                // Undecorated windowed window sized/positioned to cover the
                // selected monitor -- no monitor handle passed, so this is a
                // regular window rather than an exclusive-fullscreen surface.
                context.window = glfwCreateWindow(mode->width, mode->height, "VK", nullptr, NULL);

                if (context.window != nullptr) {
                    std::int32_t monitor_x = 0;
                    std::int32_t monitor_y = 0;
                    glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);
                    glfwSetWindowPos(context.window, monitor_x, monitor_y);
                }
                break;

            case ScreenType::windowed:
                constexpr std::int32_t default_width = 1280;
                constexpr std::int32_t default_height = 720;
                context.window = glfwCreateWindow(default_width, default_height, "VK", nullptr, NULL);
                break;
        }

        if (context.window == nullptr) {
            error("glfwCreateWindow failed");
            return false;
        }

        info("GLFW window created");

        return true;
    }

    auto create_instance(VulkanContext &context) noexcept -> bool {
        auto const volk_result = volkInitialize();

        if (volk_result != VK_SUCCESS) {
            report_vk_error("volkInitialize", volk_result);

            return false;
        }

        auto const loader_version = volkGetInstanceVersion();

        info("Vulkan loader version: {}.{}.{}", VK_VERSION_MAJOR(loader_version), VK_VERSION_MINOR(loader_version),
             VK_VERSION_PATCH(loader_version));

        if (loader_version < VK_API_VERSION_1_3) {
            error("Vulkan 1.3 is required for "
                  "core synchronization2");

            return false;
        }

        constexpr auto requested_version = VK_API_VERSION_1_4;

        std::uint32_t extension_count = 0;

        auto const *required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);

        if (required_extensions == nullptr || extension_count == 0) {
            error("glfwGetRequiredInstanceExtensions "
                  "returned no extensions");

            return false;
        }

        std::vector<char const *> instance_extensions(required_extensions, required_extensions + extension_count);

        auto validation_enabled = enable_validation;
        info("Validation is {}", validation_enabled ? "on" : "off");

        if (validation_enabled && !validation_layer_available()) {
            warn("{} is unavailable; validation "
                 "is disabled",
                 validation_layers.front());

            validation_enabled = false;
        }

        if (validation_enabled) {
            instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        auto const debug_create_info = make_debug_messenger_create_info();

        VkApplicationInfo const application_info{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = nullptr,
                .pApplicationName = "glfw-vulkan-test",
                .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
                .pEngineName = "none",
                .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
                .apiVersion = requested_version,
        };

        VkInstanceCreateInfo const create_info{
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = validation_enabled ? &debug_create_info : nullptr,
                .flags = 0,
                .pApplicationInfo = &application_info,
                .enabledLayerCount = validation_enabled ? static_cast<std::uint32_t>(validation_layers.size()) : 0,
                .ppEnabledLayerNames = validation_enabled ? validation_layers.data() : nullptr,
                .enabledExtensionCount = static_cast<std::uint32_t>(instance_extensions.size()),
                .ppEnabledExtensionNames = instance_extensions.data(),
        };

        auto const result = vkCreateInstance(&create_info, nullptr, &context.instance);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateInstance", result);

            return false;
        }

        volkLoadInstance(context.instance);

        if (validation_enabled) {
            auto const debug_result = vkCreateDebugUtilsMessengerEXT(context.instance, &debug_create_info, nullptr,
                                                                     &context.debug_messenger);

            if (debug_result != VK_SUCCESS) {
                report_vk_error("vkCreateDebugUtilsMessengerEXT", debug_result);

                return false;
            }

            info("Vulkan validation enabled");
        }

        info("Vulkan instance created");

        return true;
    }

    auto create_surface(VulkanContext &context) noexcept -> bool {
        auto const result = glfwCreateWindowSurface(context.instance, context.window, nullptr, &context.surface);

        if (result != VK_SUCCESS) {
            report_vk_error("glfwCreateWindowSurface", result);

            return false;
        }

        info("GLFW Vulkan surface created");

        return true;
    }

    auto find_queue_families(VkPhysicalDevice physical_device, VkSurfaceKHR surface) noexcept -> QueueFamilies {
        std::uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, properties.data());

        QueueFamilies result{};
        for (std::uint32_t index = 0; index < queue_family_count; ++index) {
            auto const supports_graphics = (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;

            if (supports_graphics) {
                result.graphics = index;
            }

            VkBool32 supports_present = VK_FALSE;

            auto const present_result =
                    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, surface, &supports_present);

            if (present_result != VK_SUCCESS) {
                report_vk_error("vkGetPhysicalDeviceSurfaceSupportKHR", present_result);

                continue;
            }

            if (supports_present == VK_TRUE) {
                result.present = index;
            }

            if (result.complete()) {
                break;
            }
        }

        return result;
    }

    auto supports_device_extension(VkPhysicalDevice physical_device, std::string_view required_extension) noexcept
            -> bool {
        std::uint32_t extension_count = 0;
        auto result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateDeviceExtensionProperties(count)", result);

            return false;
        }

        std::vector<VkExtensionProperties> extensions(extension_count);
        result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateDeviceExtensionProperties(list)", result);

            return false;
        }

        return std::ranges::any_of(extensions, [required_extension](VkExtensionProperties const &extension) {
            return required_extension == extension.extensionName;
        });
    }

    auto rate_device_type(VkPhysicalDeviceType type) noexcept -> int {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return 500;

            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return 400;

            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return 300;

            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return 200;

            default:
                return 100;
        }
    }

    auto select_physical_device(VulkanContext &context) noexcept -> bool {
        std::uint32_t physical_device_count = 0;
        auto result = vkEnumeratePhysicalDevices(context.instance, &physical_device_count, nullptr);
        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumeratePhysicalDevices(count)", result);
            return false;
        }

        if (physical_device_count == 0) {
            error("No Vulkan physical devices found");
            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        result = vkEnumeratePhysicalDevices(context.instance, &physical_device_count, physical_devices.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumeratePhysicalDevices(list)", result);
            return false;
        }

        auto best_score = -1;

        for (auto const physical_device: physical_devices) {
            VkPhysicalDeviceProperties properties{};

            vkGetPhysicalDeviceProperties(physical_device, &properties);

            if (properties.apiVersion < VK_API_VERSION_1_3) {
                continue;
            }

            VkPhysicalDeviceVulkan12Features vulkan12_features{};
            vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            vulkan12_features.pNext = nullptr;

            VkPhysicalDeviceVulkan13Features vulkan13_features{};
            vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            vulkan13_features.pNext = &vulkan12_features;

            VkPhysicalDeviceFeatures2 features2{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                    .pNext = &vulkan13_features,
                    .features = {},
            };

            vkGetPhysicalDeviceFeatures2(physical_device, &features2);

            if (vulkan12_features.bufferDeviceAddress != VK_TRUE || vulkan13_features.synchronization2 != VK_TRUE ||
                vulkan13_features.dynamicRendering != VK_TRUE) {
                continue;
            }

            if (!supports_device_extension(physical_device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                continue;
            }

            auto const queue_families = find_queue_families(physical_device, context.surface);

            if (!queue_families.complete()) {
                continue;
            }

            auto const score = rate_device_type(properties.deviceType);

            if (score > best_score) {
                best_score = score;
                context.physical_device = physical_device;
                context.queue_families = queue_families;
            }
        }

        if (context.physical_device == VK_NULL_HANDLE) {
            error("No suitable Vulkan physical "
                  "device found");
            return false;
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);
        info("Selected physical device: {}", properties.deviceName);

        // VK_EXT_shader_object is optional: not every target GPU implements
        // it yet (e.g. some Intel iGPUs), so its absence must not disqualify
        // an otherwise-suitable device. create_device() only enables the
        // extension/features below when this is true, and the renderer
        // falls back to VkPipeline per pipeline registration otherwise --
        // see PipelineRegisterInfo::use_shader_objects.
        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extended_dynamic_state3_features{};
        extended_dynamic_state3_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        extended_dynamic_state3_features.pNext = nullptr;

        VkPhysicalDeviceShaderObjectFeaturesEXT shader_object_features{};
        shader_object_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        shader_object_features.pNext = &extended_dynamic_state3_features;

        VkPhysicalDeviceFeatures2 shader_object_query{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &shader_object_features,
                .features = {},
        };

        vkGetPhysicalDeviceFeatures2(context.physical_device, &shader_object_query);

        context.shader_objects_supported =
                shader_object_features.shaderObject == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3ColorBlendEnable == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3ColorBlendEquation == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3ColorWriteMask == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3RasterizationSamples == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3SampleMask == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3AlphaToCoverageEnable == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3PolygonMode == VK_TRUE &&
                extended_dynamic_state3_features.extendedDynamicState3LogicOpEnable == VK_TRUE &&
                supports_device_extension(context.physical_device, VK_EXT_SHADER_OBJECT_EXTENSION_NAME) &&
                supports_device_extension(context.physical_device, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME) &&
                supports_device_extension(context.physical_device, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);

        info("VK_EXT_shader_object support: {}",
             context.shader_objects_supported ? "yes" : "no (falling back to VkPipeline)");
        return true;
    }

    auto create_device(VulkanContext &context) noexcept -> bool {
        constexpr float queue_priority = 1.0F;

        std::array<std::uint32_t, 2> queue_family_indices{
                context.queue_families.graphics,
                context.queue_families.present,
        };

        auto const unique_end = std::unique(queue_family_indices.begin(), queue_family_indices.end());
        auto const queue_count = static_cast<std::size_t>(unique_end - queue_family_indices.begin());

        std::array<VkDeviceQueueCreateInfo, 2> queue_create_infos{};
        for (std::size_t index = 0; index < queue_count; ++index) {
            queue_create_infos[index] = VkDeviceQueueCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queueFamilyIndex = queue_family_indices[index],
                    .queueCount = 1,
                    .pQueuePriorities = &queue_priority,
            };
        }

        std::vector<char const *> device_extensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_EXT_MESH_SHADER_EXTENSION_NAME,
        };

        if (context.shader_objects_supported) {
            device_extensions.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
            device_extensions.push_back(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
            device_extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        }

        VkPhysicalDeviceVulkan11Features vulkan11_features{};
        vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11_features.pNext = nullptr;
        vulkan11_features.shaderDrawParameters = VK_TRUE;
        vulkan11_features.multiview = VK_TRUE;

        VkPhysicalDeviceVulkan12Features vulkan12_features{};
        vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12_features.pNext = &vulkan11_features;
        vulkan12_features.bufferDeviceAddress = VK_TRUE;
        vulkan12_features.scalarBlockLayout = VK_TRUE;
        vulkan12_features.runtimeDescriptorArray = VK_TRUE;
        vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vulkan12_features.hostQueryReset = VK_TRUE;
        vulkan12_features.shaderFloat16 = VK_TRUE;

        VkPhysicalDeviceVulkan13Features vulkan13_features{};
        vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13_features.pNext = &vulkan12_features;
        vulkan13_features.synchronization2 = VK_TRUE;
        vulkan13_features.dynamicRendering = VK_TRUE;
        vulkan13_features.maintenance4 = VK_TRUE;
        vulkan13_features.shaderDemoteToHelperInvocation = VK_TRUE;

        VkPhysicalDeviceVulkan14Features vulkan14_features{};
        vulkan14_features.pNext = &vulkan13_features;
        vulkan14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        vulkan14_features.maintenance5 = VK_TRUE;

        VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
        mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        mesh_shader_features.pNext = &vulkan14_features;
        mesh_shader_features.taskShader = VK_TRUE;
        mesh_shader_features.meshShader = VK_TRUE;

        VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertex_input_dynamic_state_features{};
        vertex_input_dynamic_state_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
        vertex_input_dynamic_state_features.pNext = &mesh_shader_features;
        vertex_input_dynamic_state_features.vertexInputDynamicState = VK_TRUE;

        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extended_dynamic_state3_features{};
        extended_dynamic_state3_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        extended_dynamic_state3_features.pNext = &vertex_input_dynamic_state_features;
        extended_dynamic_state3_features.extendedDynamicState3ColorBlendEnable = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3ColorBlendEquation = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3ColorWriteMask = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3RasterizationSamples = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3SampleMask = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3AlphaToCoverageEnable = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3PolygonMode = VK_TRUE;
        extended_dynamic_state3_features.extendedDynamicState3LogicOpEnable = VK_TRUE;

        VkPhysicalDeviceShaderObjectFeaturesEXT shader_object_features{};
        shader_object_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        shader_object_features.pNext = &extended_dynamic_state3_features;
        shader_object_features.shaderObject = VK_TRUE;

        void const *feature_chain = context.shader_objects_supported
                                            ? static_cast<void const *>(&shader_object_features)
                                            : static_cast<void const *>(&mesh_shader_features);

        VkPhysicalDeviceFeatures enabled_features{};
        enabled_features.multiDrawIndirect = VK_TRUE;
        enabled_features.samplerAnisotropy = VK_TRUE;
        enabled_features.fillModeNonSolid = VK_TRUE;
        enabled_features.wideLines = VK_TRUE;
        enabled_features.pipelineStatisticsQuery = VK_TRUE;
        enabled_features.shaderInt16 = VK_TRUE;
        enabled_features.drawIndirectFirstInstance = VK_TRUE;

        VkDeviceCreateInfo const create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = feature_chain,
                .flags = 0,
                .queueCreateInfoCount = static_cast<std::uint32_t>(queue_count),
                .pQueueCreateInfos = queue_create_infos.data(),
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = nullptr,
                .enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size()),
                .ppEnabledExtensionNames = device_extensions.data(),
                .pEnabledFeatures = &enabled_features,
        };

        auto const result = vkCreateDevice(context.physical_device, &create_info, nullptr, &context.device);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateDevice", result);
            return false;
        }

        volkLoadDevice(context.device);
        vkGetDeviceQueue(context.device, context.queue_families.graphics, 0, &context.graphics_queue);
        vkGetDeviceQueue(context.device, context.queue_families.present, 0, &context.present_queue);

        if (context.graphics_queue == VK_NULL_HANDLE || context.present_queue == VK_NULL_HANDLE) {
            error("One or more Vulkan queues are null");
            return false;
        }

        info("Logical Vulkan device created");

        VkCommandPoolCreateInfo command_pool_create_info{};
        command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_create_info.queueFamilyIndex = context.queue_families.graphics;
        command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(context.device, &command_pool_create_info, nullptr, &context.one_time_pool) !=
            VK_SUCCESS) {
            error("Could not allocate a one time command pool;");
            return false;
        }

        VkCommandBufferAllocateInfo command_buffer_allocate_info{};
        command_buffer_allocate_info.commandPool = context.one_time_pool;
        command_buffer_allocate_info.commandBufferCount =
                static_cast<std::uint32_t>(context.one_time_command_buffers.size());
        command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        if (vkAllocateCommandBuffers(context.device, &command_buffer_allocate_info,
                                     context.one_time_command_buffers.data()) != VK_SUCCESS) {
            error("Could not allocate the command buffers");
            return false;
        }

        return true;
    }

    auto create_allocator(VulkanContext &context) noexcept -> bool {
        VmaAllocatorCreateInfo create_info{

                .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                         VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                         VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT | VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT,
                .physicalDevice = context.physical_device,
                .device = context.device,
                .preferredLargeHeapBlockSize = 0,
                .pAllocationCallbacks = nullptr,
                .pDeviceMemoryCallbacks = nullptr,
                .pHeapSizeLimit = nullptr,
                .pVulkanFunctions = nullptr,
                .instance = context.instance,
                .vulkanApiVersion = VK_API_VERSION_1_4,
                .pTypeExternalMemoryHandleTypes = nullptr,
        };

        VmaVulkanFunctions allocator_functions{};
        auto const import_result = vmaImportVulkanFunctionsFromVolk(&create_info, &allocator_functions);
        if (import_result != VK_SUCCESS) {
            error("Could not initialise VMA");
            return false;
        }

        create_info.pVulkanFunctions = &allocator_functions;
        auto const result = vmaCreateAllocator(&create_info, &context.allocator);

        if (result != VK_SUCCESS) {
            error("Could not create Vulkan allocator");
            return false;
        }

        return true;
    }

    auto create_swapchain(VulkanContext &context) noexcept -> bool {
        int framebuffer_width = 0;
        int framebuffer_height = 0;

        glfwGetFramebufferSize(context.window, &framebuffer_width, &framebuffer_height);
        if (framebuffer_width <= 0 || framebuffer_height <= 0) {
            error("Cannot create a swapchain for "
                  "a zero-sized framebuffer");

            return false;
        }

        context.framebuffer_width.store(framebuffer_width, std::memory_order_relaxed);
        context.framebuffer_height.store(framebuffer_height, std::memory_order_relaxed);

        return context.swapchain.initialize(SwapchainCreateInfo{
                .physical_device = context.physical_device,
                .device = context.device,
                .surface = context.surface,
                .graphics_queue = context.graphics_queue,
                .present_queue = context.present_queue,
                .graphics_queue_family = context.queue_families.graphics,
                .present_queue_family = context.queue_families.present,
                .framebuffer_extent =
                        VkExtent2D{
                                .width = static_cast<std::uint32_t>(framebuffer_width),
                                .height = static_cast<std::uint32_t>(framebuffer_height),
                        },
                .vsync = true,
        });
    }

    auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool {
        return initialize_glfw(context, screen_type) && create_instance(context) && create_surface(context) &&
               select_physical_device(context) && create_device(context) && create_allocator(context) &&
               create_swapchain(context);
    }

    auto submit_scene(Application &application) -> std::expected<void, RendererError> {
        auto &registry = application.active_scene->get_registry();

        auto view = registry.view<Components::Transform const, Components::Model const>();

        for (auto [entity, transform, model]: view.each()) {
            auto const *override_component = registry.try_get<Components::MaterialOverride const>(entity);
            auto const material_override =
                    override_component != nullptr ? override_component->material : MaterialHandle{};

            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto result = application.renderer->submit_model(model.model, world_transform, material_override);

            if (!result) {
                error("Could not submit scene object (model index {}): {}", model.model.index,
                      describe(result.error()));
            }
        }

        auto point_light_view = registry.view<Components::Transform const, Components::PointLight const>();

        for (auto [entity, transform, light]: point_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);

            auto result = application.renderer->submit_point_light(Renderer::PointLight{
                    .position = glm::vec3{world_transform[3]},
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
            });

            if (!result) {
                error("Could not submit point light: {}", describe(result.error()));
            }
        }

        auto spot_light_view = registry.view<Components::Transform const, Components::SpotLight const>();

        for (auto [entity, transform, light]: spot_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto const direction = glm::normalize(glm::mat3{world_transform} * glm::vec3{0.0F, -1.0F, 0.0F});

            auto result = application.renderer->submit_spot_light(Renderer::SpotLight{
                    .position = glm::vec3{world_transform[3]},
                    .direction = direction,
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
                    .inner_cone_degrees = light.inner_cone_degrees,
                    .outer_cone_degrees = light.outer_cone_degrees,
            });

            if (!result) {
                error("Could not submit spot light: {}", describe(result.error()));
            }
        }

        return {};
    }

    auto initialize_application(VulkanContext &context, Application &application) noexcept -> bool {
        auto renderer_result = application.renderer->initialize(RendererCreateInfo{
                .extent = context.swapchain.extent(),
                .geometry_capacity = 256UZ * 1024UZ * 1024UZ,
                .material_capacity = 4096,
                .mesh_capacity = 4096,
                .model_capacity = 1024,
                .pipeline_capacity = 1024,
                .swapchain_format = context.swapchain.format(),
                .samples = VK_SAMPLE_COUNT_4_BIT,
        });

        if (!renderer_result) {
            error("Could not initialize renderer: {}", describe(renderer_result.error()));
            return false;
        }

        return true;
    }

    auto draw(VulkanContext &context, Application &application) noexcept -> bool {
        auto frame = context.swapchain.begin_frame();

        if (!frame) {
            switch (frame.error().kind) {
                case SwapchainBeginFrameError::Kind::recreated:
                    return true;

                case SwapchainBeginFrameError::Kind::device_lost:
                    error("The Vulkan device was lost");

                    return false;

                case SwapchainBeginFrameError::Kind::fatal_error:
                    if (frame.error().context.has_value()) {
                        error("Could not begin swapchain frame: {}", describe(*frame.error().context));
                    } else {
                        error("Could not begin swapchain frame");
                    }

                    return false;
            }

            return false;
        }

        auto frame_ok = true;
        auto submit_result = submit_scene(application);
        if (!submit_result) {
            error("Could not submit scene: {}", describe(submit_result.error()));
            frame_ok = false;
        }

        if (frame_ok) {
            application.imgui_renderer->begin_frame(gui::ImGuiFramebuffer{frame->extent, frame->format});
            {
                application.on_ui();
            }
            application.imgui_renderer->end_frame();

            auto const &active_view =
                    application.is_playing ? application.player_camera.view() : application.camera.view();
            auto const active_aspect = application.renderer->aspect(frame->frame_index);
            auto const &active_projection = application.is_playing ? application.player_camera.projection(active_aspect)
                                                                   : application.camera.projection(active_aspect);

            auto prepare_result = application.renderer->prepare_frame(
                    frame->command_buffer,
                    {
                            .view = active_view,
                            .projection = active_projection,
                            .near_clip = application.camera.near_clip(),
                            .far_clip = application.camera.far_clip(),
                            .vertical_fov_radians = glm::radians(application.camera.field_of_view_degrees()),
                            .aspect_ratio = active_aspect,
                            .time = application.elapsed_time,
                    },
                    frame->frame_index);

            if (!prepare_result) {
                error("Could not prepare renderer frame: {}", describe(prepare_result.error()));

                frame_ok = false;
            }
        }

        if (frame_ok) {
            if (auto const &timings = application.renderer->last_frame_timings(); timings.valid) {
                application.timing_x += 1.0F;

                float running_total = 0.0F;

                for (auto stage = static_cast<std::uint32_t>(RenderStage::Culling); stage < stage_count; ++stage) {
                    running_total += timings.milliseconds[stage];
                    application.timing_buffers[stage].add_point(application.timing_x, running_total);
                }
            }


            auto const &v = application.is_playing ? application.player_camera.view() : application.camera.view();
            auto const active_aspect = application.renderer->aspect(frame->frame_index);
            auto const &p = application.is_playing ? application.player_camera.projection(active_aspect)
                                                   : application.camera.projection(active_aspect);
            auto record_result = application.renderer->record_frame<ApplicationOverlayPolicy>(
                    frame->command_buffer,
                    SwapchainImage{
                            .image = frame->image,
                            .view = frame->image_view,
                            .format = frame->format,
                            .extent = frame->extent,
                    },
                    frame->frame_index, application, p * v);

            if (!record_result) {
                error("Could not record renderer frame: {}", describe(record_result.error()));

                frame_ok = false;
            }
        }

        // Always retire the frame we began, whether or not its content ended
        // up being usable -- this is what keeps image_available's
        // signal/wait balanced and in_flight's state honest, regardless of
        // which step above failed.
        //
        // Caveat: this assumes prepare_frame()/record_frame() never return
        // failure while frame->command_buffer still has an open dynamic
        // rendering scope (vkCmdBeginRendering without a matching
        // vkCmdEndRendering) -- ending a command buffer with one open is
        // itself invalid. If that assumption doesn't hold on the renderer
        // side, this needs a matching fix there too.
        auto const end_result = context.swapchain.end_frame(*frame);

        if (!frame_ok) {
            return false;
        }

        switch (end_result) {
            case SwapchainFrameResult::success:
            case SwapchainFrameResult::recreated:
                return true;

            case SwapchainFrameResult::device_lost:
                error("The Vulkan device was lost");

                return false;

            case SwapchainFrameResult::fatal_error:
                error("Could not submit or present frame");

                return false;
        }

        return false;
    }

    auto request_resize_if_needed(VulkanContext &context, int width, int height) noexcept -> void {
        if (!context.framebuffer_dirty.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        context.swapchain.request_recreate(VkExtent2D{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
        });
    }

    struct WindowData {
        VulkanContext *ctx;
        Application *app;
    };

    auto framebuffer_size_callback(GLFWwindow *window, int width, int height) noexcept -> void {
        auto *context = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->ctx;

        if (context == nullptr) {
            return;
        }

        // This callback and the render loop both run on the main thread
        // now, so there's no concurrent reader/writer to lock against --
        // a plain relaxed store is enough.
        context->framebuffer_width.store(width, std::memory_order_relaxed);
        context->framebuffer_height.store(height, std::memory_order_relaxed);
        context->framebuffer_dirty.store(true, std::memory_order_relaxed);
    }

    auto event_callback(GLFWwindow *window, int key, int, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (action == GLFW_PRESS && app->on_event(KeyPressedEvent{key, mods})) {
        }

        if (action == GLFW_RELEASE && app->on_event(KeyReleasedEvent{key, mods})) {
        }
    }

    auto mouse_button_callback(GLFWwindow *window, int button, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Play mode owns cursor capture for its whole duration (see
        // Application::play()/stop()) -- right-drag-to-look only applies to
        // the editor camera.
        if (!app->is_playing) {
            if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            } else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }

        if (action == GLFW_PRESS) {
            app->on_event(MouseButtonPressedEvent{button, mods});
        } else if (action == GLFW_RELEASE) {
            app->on_event(MouseButtonReleasedEvent{button, mods});
        }
    }

    auto cursor_position_callback(GLFWwindow *window, double x_position, double y_position) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Swallow the first callback after (re)acquiring a cursor position:
        // without a previous sample the delta would be "distance from
        // wherever the cursor happened to be", producing a large spurious
        // look jump on the first drag frame.
        if (!app->has_last_mouse_position) {
            app->last_mouse_x = x_position;
            app->last_mouse_y = y_position;
            app->has_last_mouse_position = true;

            return;
        }

        auto const delta_x = x_position - app->last_mouse_x;
        auto const delta_y = y_position - app->last_mouse_y;

        app->last_mouse_x = x_position;
        app->last_mouse_y = y_position;

        app->on_event(MouseMovedEvent{delta_x, delta_y});
    }

    auto scroll_callback(GLFWwindow *window, double x_offset, double y_offset) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app != nullptr) {
            app->on_event(MouseScrolledEvent{x_offset, y_offset});
        }
    }

    auto focus_callback(GLFWwindow *window, int focused) noexcept -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Alt-tabbing (or any focus loss) while captured would otherwise leave
        // the cursor disabled and player_controller still integrating whatever
        // stray deltas the OS/compositor delivers to an unfocused window --
        // behaviour that varies between X11 and Wayland (see the platform
        // hint in initialize_glfw). Exiting play mode on focus loss sidesteps
        // that entirely rather than trying to special-case each platform.
        if (focused == GLFW_FALSE && app->is_playing) {
            app->stop();
        }
    }


    auto install_window_callbacks(VulkanContext &context, Application &app) noexcept -> void {
        static WindowData wd{};
        wd.app = &app;
        wd.ctx = &context;
        glfwSetWindowUserPointer(context.window, &wd);

        glfwSetFramebufferSizeCallback(context.window, framebuffer_size_callback);
        glfwSetKeyCallback(context.window, event_callback);
        glfwSetMouseButtonCallback(context.window, mouse_button_callback);
        glfwSetCursorPosCallback(context.window, cursor_position_callback);
        glfwSetScrollCallback(context.window, scroll_callback);
        glfwSetWindowFocusCallback(context.window, focus_callback);
    }

    auto destroy_context(VulkanContext &context) noexcept -> void {
        context.swapchain.destroy();

        if (context.one_time_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(context.device, context.one_time_pool, nullptr);

            context.one_time_pool = VK_NULL_HANDLE;
            context.one_time_command_buffers.fill(VK_NULL_HANDLE);
        }

        if (context.allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(context.allocator);

            context.allocator = VK_NULL_HANDLE;
        }

        if (context.device != VK_NULL_HANDLE) {
            vkDestroyDevice(context.device, nullptr);

            context.device = VK_NULL_HANDLE;
        }

        if (context.surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(context.instance, context.surface, nullptr);

            context.surface = VK_NULL_HANDLE;
        }

        if (context.debug_messenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(context.instance, context.debug_messenger, nullptr);

            context.debug_messenger = VK_NULL_HANDLE;
        }

        if (context.instance != VK_NULL_HANDLE) {
            vkDestroyInstance(context.instance, nullptr);

            context.instance = VK_NULL_HANDLE;
        }

        if (context.window != nullptr) {
            glfwDestroyWindow(context.window);

            context.window = nullptr;
        }

        glfwTerminate();
    }

    // vkDeviceWaitIdle has no timeout parameter -- a genuinely non-responding
    // GPU hangs it forever with no way to interrupt the wait from this
    // thread. Running it as an async task and giving up on *waiting for it*
    // after a bound at least lets the process exit instead of hanging. Once
    // we've given up, continuing on to call more Vulkan destroy functions
    // against a device the driver may still be touching is itself unsafe, so
    // this terminates immediately rather than attempting further cleanup.
    auto wait_idle_bounded(VkDevice device, std::string_view label) noexcept -> VkResult {
        constexpr auto timeout = std::chrono::seconds{3};

        auto future = std::async(std::launch::async, [device] { return vkDeviceWaitIdle(device); });

        if (future.wait_for(timeout) != std::future_status::ready) {
            error("{}: vkDeviceWaitIdle did not return within {} -- the GPU is not responding. Exiting immediately.",
                  label, timeout);

            std::_Exit(EXIT_FAILURE);
        }

        return future.get();
    }

    auto destroy_application(VulkanContext &context, Application &application) noexcept -> void {
        if (context.device != VK_NULL_HANDLE) {
            auto const result = wait_idle_bounded(context.device, "destroy_application");

            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                report_vk_error("vkDeviceWaitIdle(application destroy)", result);
            }
        }
        application.debug_renderer.reset();
        application.imgui_renderer.reset();
        application.renderer->destroy();

        destroy_context(context);
    }
} // namespace

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    glfwPostEmptyEvent();
}


auto main(int argc, char **argv) -> int {
    info("Starting GLFW Vulkan test at {}", std::filesystem::current_path().string());

    std::signal(SIGINT, ctrl_c_handler);

    auto const screen_type = parse_screen_type(argc, argv);

    VulkanContext context{};

    if (!initialize_vulkan(context, screen_type)) {
        error("Vulkan initialization failed");

        destroy_context(context);

        return EXIT_FAILURE;
    }


    Application application{context};
    install_window_callbacks(context, application);

    if (!initialize_application(context, application)) {
        destroy_application(context, application);

        return EXIT_FAILURE;
    }

    application.on_startup();

    info("Initialization complete; close the window to exit");

    auto renderer_extent = context.swapchain.extent();
    auto last_frame_time = std::chrono::steady_clock::now();
    auto exit_code = EXIT_SUCCESS;

    while (g_running.load(std::memory_order_acquire) && context.running.load(std::memory_order_acquire) &&
           glfwWindowShouldClose(context.window) != GLFW_TRUE) {

        auto const width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const height = context.framebuffer_height.load(std::memory_order_relaxed);

        // Minimized / zero-sized framebuffer: nothing to render, so block
        // for the next event instead of busy-looping. Everywhere else we
        // poll (non-blocking), since we want to keep rendering every
        // iteration rather than waiting for input.
        if (width <= 0 || height <= 0) {
            glfwWaitEvents();
        } else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(context.window) == GLFW_TRUE) {
            break;
        }

        application.renderer->drain_event_queue();

        auto const current_width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const current_height = context.framebuffer_height.load(std::memory_order_relaxed);

        if (current_width <= 0 || current_height <= 0) {
            last_frame_time = std::chrono::steady_clock::now();
            continue;
        }

        auto const now = std::chrono::steady_clock::now();
        auto const delta_time = std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;

        application.elapsed_time += delta_time;

        application.camera.update(std::min(delta_time, 0.1F));
        application.update(delta_time);

        request_resize_if_needed(context, current_width, current_height);

        if (!draw(context, application)) {
            exit_code = EXIT_FAILURE;
            break;
        }

        auto const swapchain_extent = context.swapchain.extent();

        if (compare(swapchain_extent, renderer_extent)) {
            auto resize_result = application.renderer->resize(swapchain_extent);

            if (!resize_result) {
                error("Could not resize renderer: {}", describe(resize_result.error()));

                exit_code = EXIT_FAILURE;
                break;
            }

            renderer_extent = swapchain_extent;
        }
    }

    context.running.store(false, std::memory_order_release);

    destroy_application(context, application);

    if (exit_code == EXIT_SUCCESS) {
        info("Application exited successfully");
    }

    return exit_code;
}
