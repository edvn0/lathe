#include <csignal>
#include <random>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "allocator.hxx"
#include "config.hxx"
#include "context.hxx"
#include "editor_camera.hxx"
#include "logger.hxx"
#include "renderdoc.hxx"
#include "renderer.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "swapchain.hxx"

namespace {
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

    struct SceneTransform {
        glm::vec3 position{0.0F};
        glm::quat rotation{};
        glm::vec3 scale{1.0F};

        auto matrix() const {
            return glm::translate(glm::mat4{1.0F}, position) * glm::mat4_cast(rotation) *
                   glm::scale(glm::mat4{1.0F}, scale);
        }
    };

    struct SceneObject {
        ModelHandle model{};
        SceneTransform transform{};
    };

    struct RenderScene {
        std::vector<SceneObject> objects;
    };

    struct KeyPressedEvent {
        std::int32_t key{};
        std::int32_t modifiers{};
    };

    struct KeyReleasedEvent {
        std::int32_t key{};
        std::int32_t modifiers{};
    };

    // Raw cursor delta in pixels since the previous callback -- not an
    // absolute position. Application decides whether it counts as a
    // "look" based on whether a drag is currently active.
    struct MouseMovedEvent {
        double delta_x{};
        double delta_y{};
    };

    struct MouseScrolledEvent {
        double delta_x{};
        double delta_y{};
    };

    struct MouseButtonPressedEvent {
        std::int32_t button{};
        std::int32_t modifiers{};
    };

    struct MouseButtonReleasedEvent {
        std::int32_t button{};
        std::int32_t modifiers{};
    };

    struct Application {
        explicit Application(VulkanContext &context) noexcept : renderer{context} {
            std::array const shader_directories{
                    std::filesystem::path{"assets/shaders"},
            };
            if (!shader_watcher_.start(renderer, shader_directories)) {
                error("Shader hot-reload watcher failed to start -- shaders will not live-reload this run");
            }
        }
        ~Application() { shader_watcher_.stop(); }

        Renderer renderer;
        RenderScene scene;
        ShaderHotReloadWatcher shader_watcher_;

        EditorCamera camera;

        // Right-mouse-drag look state. Cursor capture itself (hide + relative
        // mode) is toggled directly in mouse_button_callback since that's
        // where the GLFWwindow handle is already available.
        bool mouse_dragging = false;

        double last_mouse_x = 0.0;
        double last_mouse_y = 0.0;
        bool has_last_mouse_position = false;

        auto on_startup() -> void {
            renderer.queue_render_thread_event([this] { this->recreate_entities(); });
        }

        auto on_event(KeyPressedEvent ev) -> bool {
            if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL) {
                renderer.queue_render_thread_event([this] { this->recreate_entities(); });
            }
            camera.on_key_pressed(ev.key);
            return true;
        }

        auto on_event(KeyReleasedEvent ev) -> bool {
            camera.on_key_released(ev.key);

            return true;
        }

        auto on_event(MouseMovedEvent ev) -> bool {
            camera.on_mouse_moved(static_cast<float>(ev.delta_x), static_cast<float>(ev.delta_y), mouse_dragging);

            return true;
        }

        auto on_event(MouseScrolledEvent ev) -> bool {
            camera.on_mouse_scrolled(static_cast<float>(ev.delta_y));

            return true;
        }

        auto on_event(MouseButtonPressedEvent ev) -> bool {
            if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
                mouse_dragging = true;
            }

            return true;
        }

        auto on_event(MouseButtonReleasedEvent ev) -> bool {
            if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
                mouse_dragging = false;
            }

            return true;
        }

        auto recreate_entities() -> void {

            if (auto could_wait = renderer.wait_idle(); !could_wait.has_value()) {
                info("{}", could_wait.error().device_error.message.view());
                return;
            }

            scene.objects.clear();


            auto helmet_model = renderer.load_model("assets/models/damaged_helmet/DamagedHelmet.gltf");
            if (!helmet_model) {
                error("Could not load model: {}", helmet_model.error().type);
            }

            auto cube_model = renderer.load_model("assets/models/test_cube.glb");
            if (!cube_model) {
                error("Could not load model: {}", helmet_model.error().type);
            }

            std::random_device r;
            std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
            std::mt19937 eng(seed);
            std::uniform_real_distribution<float> urd(-20, 20);

            const auto count = 10;
            const auto other_count = 10;
            for (auto i = 0; i < count; i++) {
                for (auto j = 0; j < count; j++) {
                    for (auto k = 0; k < count; k++) {
                        scene.objects.emplace_back(*helmet_model, SceneTransform{
                                                                          .position =
                                                                                  glm::vec3{
                                                                                          urd(eng),
                                                                                          urd(eng),
                                                                                          urd(eng),
                                                                                  },
                                                                  });
                    }
                }
            }

            for (auto i = 0; i < other_count; i++) {
                for (auto j = 0; j < other_count; j++) {
                    for (auto k = 0; k < other_count; k++) {
                        scene.objects.emplace_back(*cube_model, SceneTransform{
                                                                        .position =
                                                                                glm::vec3{
                                                                                        urd(eng),
                                                                                        urd(eng),
                                                                                        urd(eng),
                                                                                },
                                                                });
                    }
                }
            }
        }
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

    auto initialize_glfw(VulkanContext &context) noexcept -> bool {
        glfwSetErrorCallback(glfw_error_callback);


        auto renderdoc = renderdoc_init();

        info("Was created via renderdoc: {}", renderdoc.is_active());
        glfwInitHint(GLFW_PLATFORM, renderdoc.is_active() ? GLFW_PLATFORM_X11 : GLFW_PLATFORM_WAYLAND);
        warn("Chosing: {} as platform.", renderdoc.is_active() ? "X11" : "Wayland");

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
        context.window = glfwCreateWindow(mode->width, mode->height, "VK", monitor, NULL);

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

        constexpr std::array device_extensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        VkPhysicalDeviceVulkan11Features vulkan11_features{};
        vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11_features.pNext = nullptr;
        vulkan11_features.shaderDrawParameters = VK_TRUE;

        VkPhysicalDeviceVulkan12Features vulkan12_features{};
        vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12_features.pNext = &vulkan11_features;
        vulkan12_features.bufferDeviceAddress = VK_TRUE;
        vulkan12_features.scalarBlockLayout = VK_TRUE;
        vulkan12_features.runtimeDescriptorArray = VK_TRUE;
        vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vulkan12_features.hostQueryReset = VK_TRUE;

        VkPhysicalDeviceVulkan13Features vulkan13_features{};
        vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13_features.pNext = &vulkan12_features;
        vulkan13_features.synchronization2 = VK_TRUE;
        vulkan13_features.dynamicRendering = VK_TRUE;

        VkPhysicalDeviceVulkan14Features vulkan14_features{};
        vulkan14_features.pNext = &vulkan13_features;
        vulkan14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        vulkan14_features.maintenance5 = VK_TRUE;

        VkPhysicalDeviceFeatures enabled_features{};
        enabled_features.multiDrawIndirect = VK_TRUE;
        enabled_features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo const create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = &vulkan14_features,
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

    auto initialize_vulkan(VulkanContext &context) noexcept -> bool {
        return initialize_glfw(context) && create_instance(context) && create_surface(context) &&
               select_physical_device(context) && create_device(context) && create_allocator(context) &&
               create_swapchain(context);
    }

    auto submit_scene(Application &application) -> std::expected<void, RendererError> {
        for (auto const &object: application.scene.objects) {
            auto result = application.renderer.submit_model(object.model, object.transform.matrix());

            if (!result) {
                return std::unexpected(result.error());
            }
        }

        return {};
    }

    auto initialize_application(VulkanContext &context, Application &application) noexcept -> bool {
        auto renderer_result = application.renderer.initialize(RendererCreateInfo{
                .extent = context.swapchain.extent(),
                .frames_in_flight = context.swapchain.frame_count(),
                .geometry_capacity = 256UZ * 1024UZ * 1024UZ,
                .material_capacity = 4096,
                .mesh_capacity = 4096,
                .model_capacity = 1024,
                .pipeline_capacity = 1024,
                .swapchain_format = context.swapchain.format(),
                .maximum_draw_count = 65536,
                .maximum_submission_count = 65536,
        });

        if (!renderer_result) {
            error("Could not initialize renderer: {} - {}", renderer_result.error().type,
                  renderer_result.error().compiler_error.diagnostics);

            return false;
        }

        return true;
    }

    auto draw(VulkanContext &context, Application &application) noexcept -> bool {
        auto frame = context.swapchain.begin_frame();

        if (!frame) {
            switch (frame.error()) {
                case SwapchainBeginFrameError::recreated:
                    return true;

                case SwapchainBeginFrameError::device_lost:
                    error("The Vulkan device was lost");

                    return false;

                case SwapchainBeginFrameError::fatal_error:
                    error("Could not begin swapchain frame");

                    return false;
            }

            return false;
        }

        auto submit_result = submit_scene(application);

        if (!submit_result) {
            error("Could not submit scene: {}", static_cast<int>(submit_result.error().type));

            return false;
        }

        auto aspect = application.renderer.aspect(frame->frame_index);
        auto prepare_result =
                application.renderer.prepare_frame(frame->command_buffer,
                                                   {
                                                           .view = application.camera.view(),
                                                           .projection = application.camera.projection(aspect),
                                                   },
                                                   frame->frame_index);

        if (!prepare_result) {
            error("Could not prepare renderer frame: {}", static_cast<int>(prepare_result.error().type));

            return false;
        }

        auto record_result = application.renderer.record_frame(frame->command_buffer,
                                                               SwapchainImage{
                                                                       .image = frame->image,
                                                                       .view = frame->image_view,
                                                                       .format = frame->format,
                                                                       .extent = frame->extent,
                                                               },

                                                               frame->frame_index);

        if (!record_result) {
            error("Could not record renderer frame: {}", static_cast<int>(record_result.error().type));

            return false;
        }

        auto const result = context.swapchain.end_frame(*frame);

        switch (result) {
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

    auto render_loop(std::stop_token stop_token, VulkanContext &context, Application &application) noexcept -> void {

        auto renderer_extent = context.swapchain.extent();
        auto last_frame_time = std::chrono::steady_clock::now();

        while (!stop_token.stop_requested() && context.running.load(std::memory_order_acquire)) {
            application.renderer.drain_event_queue();

            auto const width = context.framebuffer_width.load(std::memory_order_acquire);
            auto const height = context.framebuffer_height.load(std::memory_order_acquire);

            if (width <= 0 || height <= 0) {
                std::unique_lock lock{context.render_wake_mutex};

                context.render_wake_condition.wait(lock, [&context, &stop_token] {
                    return stop_token.stop_requested() || !context.running.load(std::memory_order_acquire) ||
                           (context.framebuffer_width.load(std::memory_order_acquire) > 0 &&
                            context.framebuffer_height.load(std::memory_order_acquire) > 0);
                });

                last_frame_time = std::chrono::steady_clock::now();
                continue;
            }

            auto const now = std::chrono::steady_clock::now();
            auto const delta_time = std::chrono::duration<float>(now - last_frame_time).count();
            last_frame_time = now;

            application.camera.update(std::min(delta_time, 0.1F));

            request_resize_if_needed(context, width, height);

            if (!draw(context, application)) {
                context.running.store(false, std::memory_order_release);

                glfwPostEmptyEvent();
                break;
            }

            auto const swapchain_extent = context.swapchain.extent();

            if (compare(swapchain_extent, renderer_extent)) {
                auto resize_result = application.renderer.resize(swapchain_extent);

                if (!resize_result) {
                    error("Could not resize renderer: {}", static_cast<int>(resize_result.error().type));

                    context.running.store(false, std::memory_order_release);

                    glfwPostEmptyEvent();
                    break;
                }

                renderer_extent = swapchain_extent;
            }
        }
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

        context->framebuffer_width.store(width, std::memory_order_release);
        context->framebuffer_height.store(height, std::memory_order_release);
        context->framebuffer_dirty.store(true, std::memory_order_release);
        context->render_wake_condition.notify_one();
    }

    auto window_refresh_callback(GLFWwindow *window) noexcept -> void {
        auto *context = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->ctx;

        if (context != nullptr) {
            context->render_wake_condition.notify_one();
        }
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

        // Capture + hide the cursor for the duration of the drag so it can't
        // leave the window mid-look; GLFW_CURSOR_DISABLED also switches to
        // unbounded virtual cursor movement, which is what we want for a
        // free-look camera rather than clamping at the screen edge.
        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

            app->on_event(MouseButtonPressedEvent{button, mods});
        } else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

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


    auto install_window_callbacks(VulkanContext &context, Application &app) noexcept -> void {
        static WindowData wd{};
        wd.app = &app;
        wd.ctx = &context;
        glfwSetWindowUserPointer(context.window, &wd);

        glfwSetFramebufferSizeCallback(context.window, framebuffer_size_callback);
        glfwSetWindowRefreshCallback(context.window, window_refresh_callback);
        glfwSetKeyCallback(context.window, event_callback);
        glfwSetMouseButtonCallback(context.window, mouse_button_callback);
        glfwSetCursorPosCallback(context.window, cursor_position_callback);
        glfwSetScrollCallback(context.window, scroll_callback);
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

    auto destroy_application(VulkanContext &context, Application &application) noexcept -> void {
        if (context.device != VK_NULL_HANDLE) {
            auto const result = vkDeviceWaitIdle(context.device);

            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                report_vk_error("vkDeviceWaitIdle(application destroy)", result);
            }
        }

        application.renderer.destroy();

        destroy_context(context);
    }
} // namespace

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    info("Sigint received, exiting gracefully.");
    glfwPostEmptyEvent();
}


auto main() -> int {
    info("Starting GLFW Vulkan test at {}", std::filesystem::current_path().string());

    std::signal(SIGINT, ctrl_c_handler);

    VulkanContext context{};

    if (!initialize_vulkan(context)) {
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

    std::jthread render_thread{[&context, &application](std::stop_token stop_token) {
        render_loop(std::move(stop_token), context, application);
    }};

    application.on_startup();
    context.render_wake_condition.notify_one();

    info("Initialization complete; close the window to exit");

    while (g_running.load(std::memory_order_acquire) && context.running.load(std::memory_order_acquire) &&
           glfwWindowShouldClose(context.window) != GLFW_TRUE) {
        glfwWaitEvents();
    }

    context.running.store(false, std::memory_order_release);
    render_thread.request_stop();
    context.render_wake_condition.notify_one();
    render_thread.join();

    auto const render_failed = glfwWindowShouldClose(context.window) != GLFW_TRUE;

    destroy_application(context, application);

    if (render_failed) {
        return EXIT_FAILURE;
    }

    info("Application exited successfully");

    return EXIT_SUCCESS;
}
