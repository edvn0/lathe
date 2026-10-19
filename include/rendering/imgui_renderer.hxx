#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <volk.h>

#include <tuple>
#include <unordered_map>
#include <vector>

#include "gpu/image.hxx"
#include "gpu/sampler.hxx"
#include "gpu/upload_ring.hxx"
#include "rendering/overlay.hxx"
#include "rendering/overlay_shader.hxx"

#include <imgui.h>

extern "C" {
struct GLFWwindow;
}

struct Renderer;
class Pipeline;

namespace gui {

    struct FontChoice {
        std::string font_path;
        float size{20.0F};
    };

    inline constexpr std::uint64_t linear_source_texture_bit = std::uint64_t{1} << 32;

    [[nodiscard]] constexpr auto linear_source_texture_id(std::uint32_t bindless_index) noexcept -> ImTextureID {
        return ImTextureID{static_cast<std::uint64_t>(bindless_index) | linear_source_texture_bit};
    }

    // Marks ImGui's own textures, which hold glyph distance fields (see imgui_sdf_font.hxx): the shader turns the
    // field into coverage instead of using it as colour.
    inline constexpr std::uint64_t font_texture_bit = std::uint64_t{1} << 34;

    // Stands for the image Renderer::set_preview_resource names. The bindless index of a transient is only known
    // once the frame graph has allocated it, which is after the UI is built, so the ImGui renderer resolves this
    // tag when it records.
    inline constexpr std::uint64_t preview_texture_bit = std::uint64_t{1} << 33;

    [[nodiscard]] constexpr auto preview_texture_id() noexcept -> ImTextureID { return ImTextureID{preview_texture_bit}; }

    class ImGuiRenderer {
    public:
        ImGuiRenderer(Renderer &, FontChoice const &);
        ~ImGuiRenderer();

        ImGuiRenderer(ImGuiRenderer &&) = delete;
        auto operator=(ImGuiRenderer &&) -> ImGuiRenderer & = delete;

        auto update_font(FontChoice const &) -> void;
        auto set_app_name(std::string_view) -> void;

        // The same three steps as every immediate-mode renderer: build the frame, record it into an overlay's
        // command buffer, finish it. `framebuffer` is the size of the target the UI is drawn into.
        auto begin_frame(VkExtent2D framebuffer) -> void;
        auto record(OverlayRecordContext const &context) -> void;
        auto end_frame() -> void;

    private:
        ImGuiRenderer(GLFWwindow *main_window, Renderer &, FontChoice const &);

        std::vector<std::byte> font_data;
        std::vector<std::byte> emoji_font_data;

        OverlayShader shader;
        UploadRing vertices;
        UploadRing indices;
        SamplerHandle sampler{};

        struct RetiredTexture {
            ImageHandle image;
            std::uint32_t frames;
        };
        std::unordered_map<ImTextureData *, ImageHandle> textures;
        std::vector<RetiredTexture> retired_textures;

        std::string config_name{"imgui.ini"};
        std::unique_ptr<std::filesystem::path> config_path;

        Renderer &renderer;

        float display_scale{1.0F};

        auto update_textures(ImDrawData const &data) -> void;
        auto draw(OverlayRecordContext const &context, BoundOverlayShader const &bound, ImDrawData const &data) -> void;
    };

}
