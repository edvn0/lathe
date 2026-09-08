#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <volk.h>

#include <tuple>
#include <vector>

#include "gpu/buffer.hxx"
#include "gpu/image.hxx"
#include "gpu/sampler.hxx"
#include "rendering/pipeline_graph_repository.hxx"

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

    using ImGuiFramebuffer = std::tuple<VkExtent2D, VkFormat>;

    // gui.slang's fragment shader unconditionally applies its own manual
    // sRGB->linear decode, matching every bindless texture ImGui normally
    // displays (font atlas, asset thumbnails, ...): those are plain UNORM
    // images, so that manual decode is the *only* decode applied before the
    // result is written to the (SRGB-format) render target, which
    // hardware-encodes it back on store -- a deliberate round trip that
    // displays the stored bytes unchanged.
    //
    // A texture that's itself SRGB-format (e.g. Renderer::viewport_target,
    // which must stay SRGB to match what the composite pass's
    // auto-encode-on-write already assumes -- see composite.slang, which
    // relies on the same trick and does no manual encoding of its own) is
    // *already* hardware-decoded to linear by the sampler read. Tag such a
    // texture's ImTextureID with this bit so ImGuiRenderer::render_draw_data
    // can tell it apart and skip the redundant manual decode -- otherwise
    // the value gets decoded twice and the image renders too dark.
    inline constexpr std::uint64_t linear_source_texture_bit = std::uint64_t{1} << 32;

    [[nodiscard]] constexpr auto linear_source_texture_id(std::uint32_t bindless_index) noexcept -> ImTextureID {
        return ImTextureID{static_cast<std::uint64_t>(bindless_index) | linear_source_texture_bit};
    }

    class ImGuiRenderer {
    public:
        ImGuiRenderer(Renderer &, FontChoice);
        ~ImGuiRenderer();

        ImGuiRenderer(ImGuiRenderer &&) = delete;
        auto operator=(ImGuiRenderer &&) -> ImGuiRenderer & = delete;

        auto update_font(FontChoice) -> void;
        auto set_app_name(std::string_view) -> void;

        auto begin_frame(ImGuiFramebuffer main_fb) -> void;

        // frame_index must match whatever index Renderer::record_frame()
        // is using this frame -- it's forwarded straight to
        // GpuResourceTable::bind() so the bindless set matches what
        // prepare_frame() populated for that frame.
        auto render(VkCommandBuffer cmd, std::uint32_t frame_index) -> void;
        auto end_frame() -> void;

        auto set_should_recompile() -> void { force_recompile_primary = true; }

    private:
        ImGuiRenderer(GLFWwindow *main_window, std::uint32_t initial_slot_count, Renderer &, FontChoice);

        struct DrawableData {
            std::unique_ptr<Buffer> vertex;
            std::unique_ptr<Buffer> index;
            std::uint32_t index_count{0};
            std::uint32_t vertex_count{0};
        };

        PipelineNodeHandle main_pipeline{};
        SamplerHandle sampler{};
        ImageHandle font_texture{};

        std::string config_name{"imgui.ini"};
        std::unique_ptr<std::filesystem::path> config_path;

        Renderer &renderer;

        float display_scale{1.0F};

        std::vector<DrawableData> drawables{};
        std::uint32_t slots_per_frame{0};
        std::uint32_t slot_cursor{0};
        std::uint32_t frame_cursor{0};

        bool force_recompile_primary{false};

        auto render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, ShaderObjectSet const &pipeline,
                              std::uint32_t frame_index) -> void;
        auto acquire_draw_slot() -> DrawableData &;
    };

} // namespace gui
