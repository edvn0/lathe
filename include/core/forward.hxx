#pragma once

#include <cstdint>

struct VulkanContext;

class ImageStorage;
class SamplerStorage;
struct MaterialStorage;

namespace debug_draw {
    class DebugRenderer;
}

struct Runtime;
struct Application;

struct Renderer;

namespace Components {
    struct Transform;
    struct MaterialOverride;
}

class ScreenshotCapture;
enum class ScreenshotSource : std::uint8_t;
