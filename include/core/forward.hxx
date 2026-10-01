#pragma once

#include <cstdint>

struct VulkanContext;

class ImageStorage;
class SamplerStorage;
struct MaterialStorage;

namespace debug_draw {
    class DebugRenderer;
}

struct Application;

struct Renderer;

namespace Components {
    struct Transform;
    struct MaterialOverride;
} // namespace Components

class ScreenshotCapture;
enum class ScreenshotSource : std::uint8_t;
