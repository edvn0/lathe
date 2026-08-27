#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <volk.h>

#include <tuple>
#include <vector>

#include "buffer.hxx"
#include "image.hxx"
#include "pipeline_graph_repository.hxx"
#include "sampler.hxx"

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

    private:
        auto render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, Pipeline const &pipeline, std::uint32_t frame_index)
                -> void;
        auto acquire_draw_slot() -> DrawableData &;
    };

} // namespace gui
