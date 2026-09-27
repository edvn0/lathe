#pragma once

// Stable, heavy headers only; project headers change too often.

// C++ standard library
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Vulkan loader (VK_NO_PROTOTYPES is defined by the build)
#include <volk.h>

// Windowing (GLFW_INCLUDE_NONE is defined by the build)
#include <GLFW/glfw3.h>

// Math
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// ECS
#include <entt/entt.hpp>

// Thread pool
#include <BS_thread_pool.hpp>

// Profiling (no-ops without TRACY_ENABLE)
#include <tracy/Tracy.hpp>
