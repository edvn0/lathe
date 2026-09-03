#include "vulkan_bootstrap.hxx"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <vector>

#include "gpu/context.hxx"
#include "core/logger.hxx"
#include "gpu/renderdoc.hxx"

namespace {

#ifndef NDEBUG
    constexpr bool enable_validation = true;

    constexpr std::array validation_layers{
            "VK_LAYER_KHRONOS_validation",
    };
#else
    constexpr bool enable_validation = false;

    constexpr std::array<const char *, 0> validation_layers{};
#endif


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

    auto calibrateable_time_domains(VkPhysicalDevice physical_device) noexcept -> std::vector<VkTimeDomainEXT> {
        std::uint32_t domain_count = 0;
        auto result = vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical_device, &domain_count, nullptr);

        if (result != VK_SUCCESS || domain_count == 0) {
            return {};
        }

        std::vector<VkTimeDomainEXT> domains(domain_count);
        result = vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical_device, &domain_count, domains.data());

        if (result != VK_SUCCESS) {
            return {};
        }

        return domains;
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

        // Tracy's "host query" Vulkan context (see host_query_context.hxx)
        // needs VK_EXT_calibrated_timestamps plus a calibrateable device/host
        // time domain pair -- absent either, HostQueryContext::initialize
        // falls back to a calibrated or plain Tracy Vulkan context instead.
        context.calibrated_timestamps_supported =
                supports_device_extension(context.physical_device, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);

        if (context.calibrated_timestamps_supported) {
            auto const domains = calibrateable_time_domains(context.physical_device);

#ifdef _WIN32
            constexpr auto host_time_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
            constexpr auto host_time_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#endif

            context.host_calibrated_timestamps_supported =
                    std::ranges::find(domains, VK_TIME_DOMAIN_DEVICE_EXT) != domains.end() &&
                    std::ranges::find(domains, host_time_domain) != domains.end();
        }

        info("Calibrated timestamps support: {} (host-calibrated: {})",
             context.calibrated_timestamps_supported ? "yes" : "no",
             context.host_calibrated_timestamps_supported ? "yes" : "no");

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

        if (context.calibrated_timestamps_supported) {
            device_extensions.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
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

    auto create_host_query_context(VulkanContext &context) noexcept -> bool {
        context.host_query_context.initialize(context);

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


} // namespace

auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void {
    error("{} failed: {} ({})", operation, vk_result_name(result), static_cast<int>(result));
}

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

auto initialize_vulkan(VulkanContext &context, ScreenType screen_type) noexcept -> bool {
    return initialize_glfw(context, screen_type) && create_instance(context) && create_surface(context) &&
           select_physical_device(context) && create_device(context) && create_host_query_context(context) &&
           create_allocator(context) && create_swapchain(context);
}
