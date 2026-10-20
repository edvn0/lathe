#include <volk.h>

#include "rendering/imgui_renderer.hxx"
#include "core/paths.hxx"
#include "core/resources.hxx"

#include <backends/imgui_impl_glfw.h>

#include <imgui.h>

#include <ImGuizmo.h>
#include <array>
#include <bit>
#include <filesystem>
#include <implot.h>
#include <unordered_map>
#include <utility>

#include "IconsFontAwesome6.h"
#include "fa-solid-900.h"

#include "core/error_describe.hxx"
#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/imgui_sdf_font.hxx"
#include "rendering/renderer.hxx"

namespace gui {

    namespace {
        struct PC {
            std::array<float, 4> lrtb{};
            VkDeviceAddress vb;
            std::uint32_t base_vertex;
            std::uint32_t texture_id;
            std::uint32_t sampler_id{0};
            std::uint32_t already_linear{0};
            std::uint32_t sdf_text{0};
        };

        // Frames a replaced atlas image is kept for, so the frames still in flight can finish sampling it.
        constexpr std::uint32_t retired_texture_frames = 4;

        auto apply_dark_theme() -> void {
            ImGui::StyleColorsDark();
            ImGuiStyle &style = ImGui::GetStyle();

            style.WindowPadding = {8.F, 8.F};
            style.FramePadding = {6.F, 4.F};
            style.CellPadding = {6.F, 4.F};
            style.ItemSpacing = {8.F, 4.F};
            style.ItemInnerSpacing = {4.F, 4.F};
            style.IndentSpacing = 16.F;
            style.ScrollbarSize = 12.F;
            style.GrabMinSize = 8.F;

            style.WindowRounding = 4.F;
            style.ChildRounding = 4.F;
            style.FrameRounding = 3.F;
            style.PopupRounding = 4.F;
            style.ScrollbarRounding = 6.F;
            style.GrabRounding = 3.F;
            style.TabRounding = 4.F;

            style.WindowBorderSize = 1.F;
            style.ChildBorderSize = 1.F;
            style.PopupBorderSize = 1.F;
            style.FrameBorderSize = 0.F;
            style.TabBorderSize = 0.F;

            auto *c = style.Colors;
            c[ImGuiCol_Text] = {0.82F, 0.82F, 0.82F, 1.00F};
            c[ImGuiCol_TextDisabled] = {0.42F, 0.42F, 0.44F, 1.00F};
            c[ImGuiCol_WindowBg] = {0.13F, 0.13F, 0.14F, 1.00F};
            c[ImGuiCol_ChildBg] = {0.10F, 0.10F, 0.11F, 1.00F};
            c[ImGuiCol_PopupBg] = {0.11F, 0.11F, 0.12F, 0.96F};
            c[ImGuiCol_Border] = {0.25F, 0.25F, 0.27F, 0.60F};
            c[ImGuiCol_BorderShadow] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_FrameBg] = {0.18F, 0.18F, 0.20F, 1.00F};
            c[ImGuiCol_FrameBgHovered] = {0.24F, 0.24F, 0.26F, 1.00F};
            c[ImGuiCol_FrameBgActive] = {0.28F, 0.28F, 0.31F, 1.00F};
            c[ImGuiCol_TitleBg] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TitleBgActive] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TitleBgCollapsed] = {0.09F, 0.09F, 0.10F, 0.75F};
            c[ImGuiCol_MenuBarBg] = {0.11F, 0.11F, 0.12F, 1.00F};
            c[ImGuiCol_ScrollbarBg] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_ScrollbarGrab] = {0.28F, 0.28F, 0.30F, 1.00F};
            c[ImGuiCol_ScrollbarGrabHovered] = {0.34F, 0.34F, 0.37F, 1.00F};
            c[ImGuiCol_ScrollbarGrabActive] = {0.40F, 0.40F, 0.44F, 1.00F};
            c[ImGuiCol_CheckMark] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_SliderGrab] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_SliderGrabActive] = {0.46F, 0.54F, 0.80F, 1.00F};
            c[ImGuiCol_Button] = {0.24F, 0.24F, 0.27F, 1.00F};
            c[ImGuiCol_ButtonHovered] = {0.26F, 0.59F, 0.98F, 0.55F};
            c[ImGuiCol_ButtonActive] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_Header] = {0.26F, 0.59F, 0.98F, 0.25F};
            c[ImGuiCol_HeaderHovered] = {0.26F, 0.59F, 0.98F, 0.50F};
            c[ImGuiCol_HeaderActive] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_Separator] = {0.25F, 0.25F, 0.27F, 0.60F};
            c[ImGuiCol_SeparatorHovered] = {0.26F, 0.59F, 0.98F, 0.60F};
            c[ImGuiCol_SeparatorActive] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_ResizeGrip] = {0.26F, 0.59F, 0.98F, 0.20F};
            c[ImGuiCol_ResizeGripHovered] = {0.26F, 0.59F, 0.98F, 0.67F};
            c[ImGuiCol_ResizeGripActive] = {0.26F, 0.59F, 0.98F, 0.95F};
            c[ImGuiCol_Tab] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TabHovered] = {0.30F, 0.30F, 0.34F, 1.00F};
            c[ImGuiCol_TabActive] = {0.20F, 0.20F, 0.23F, 1.00F};
            c[ImGuiCol_TabUnfocused] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TabUnfocusedActive] = {0.14F, 0.14F, 0.16F, 1.00F};
            c[ImGuiCol_DockingPreview] = {0.26F, 0.59F, 0.98F, 0.60F};
            c[ImGuiCol_DockingEmptyBg] = {0.10F, 0.10F, 0.11F, 1.00F};
            c[ImGuiCol_PlotLines] = {0.61F, 0.61F, 0.61F, 1.00F};
            c[ImGuiCol_PlotLinesHovered] = {1.00F, 0.43F, 0.35F, 1.00F};
            c[ImGuiCol_PlotHistogram] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_PlotHistogramHovered] = {1.00F, 0.43F, 0.35F, 1.00F};
            c[ImGuiCol_TableHeaderBg] = {0.13F, 0.13F, 0.15F, 1.00F};
            c[ImGuiCol_TableBorderStrong] = {0.25F, 0.25F, 0.27F, 1.00F};
            c[ImGuiCol_TableBorderLight] = {0.20F, 0.20F, 0.22F, 1.00F};
            c[ImGuiCol_TableRowBg] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_TableRowBgAlt] = {1.00F, 1.00F, 1.00F, 0.03F};
            c[ImGuiCol_TextSelectedBg] = {0.26F, 0.59F, 0.98F, 0.35F};
            c[ImGuiCol_DragDropTarget] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_NavHighlight] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_NavWindowingHighlight] = {1.00F, 1.00F, 1.00F, 0.70F};
            c[ImGuiCol_NavWindowingDimBg] = {0.80F, 0.80F, 0.80F, 0.20F};
            c[ImGuiCol_ModalWindowDimBg] = {0.10F, 0.10F, 0.10F, 0.45F};
        }
    }

    ImGuiRenderer::ImGuiRenderer(Renderer &r, FontChoice const &font) :
        ImGuiRenderer(r.context().window, r, font) {}

    ImGuiRenderer::ImGuiRenderer(GLFWwindow *w, Renderer &r, FontChoice const &font) :
        shader{r, OverlayShaderInfo{.source = "assets/shaders/gui.slang",
                                    .blending = true,
                                    .debug_name = "gui.pipeline"}},
        vertices{UploadRing::Info{.name = "imgui.vertices"}},
        indices{UploadRing::Info{.name = "imgui.indices", .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT}},
        renderer(r) {

        std::ignore = ImGui::CreateContext();
        std::ignore = ImPlot::CreateContext();
        apply_dark_theme();

        ImGuiIO &io = ImGui::GetIO();
        config_name = state_path("imgui.ini").absolute().string();
        io.IniFilename = config_name.c_str();
        io.BackendRendererName = "imgui-custom-vulkan";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
        // Lines drawn from atlas texels would read the glyph field's alpha as a distance.
        ImGui::GetStyle().AntiAliasedLinesUseTex = false;

        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports;
            io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

            ImGuiStyle &style = ImGui::GetStyle();
            style.WindowRounding = 0.0F;
            style.Colors[ImGuiCol_WindowBg].w = 1.0F;
        }

        update_font(font);
        ImGui_ImplGlfw_InitForVulkan(w, true);
    }

    ImGuiRenderer::~ImGuiRenderer() {
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyPlatformWindows();

        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }

    auto ImGuiRenderer::begin_frame(VkExtent2D const framebuffer) -> void {
        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(framebuffer.width) / display_scale,
                                static_cast<float>(framebuffer.height) / display_scale);
        io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
    }

    auto ImGuiRenderer::end_frame() -> void {
        ImGui::EndFrame();
        ImGui::Render();

        if (auto &io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
        }
    }

    auto ImGuiRenderer::record(OverlayRecordContext const &context) -> void {
        auto const *const data = ImGui::GetDrawData();

        if (data == nullptr || data->TotalIdxCount == 0) {
            return;
        }

        auto const bound = shader.bind(context);
        if (!bound) {
            return;
        }

        draw(context, *bound, *data);
    }

    auto ImGuiRenderer::draw(OverlayRecordContext const &context, BoundOverlayShader const &bound,
                             ImDrawData const &data) -> void {
        auto &gpu = renderer.context();
        auto const frame = context.frame_index;

        update_textures(data);

        for (auto *ring: {&vertices, &indices}) {
            if (auto const begun = ring->begin_frame(gpu, frame); !begun) {
                error("[ImGui] Could not prepare an upload buffer: {}", begun.error().message);
                return;
            }
        }

        auto const fb_width = data.DisplaySize.x * data.FramebufferScale.x;
        auto const fb_height = data.DisplaySize.y * data.FramebufferScale.y;

        bound.set_viewport(VkViewport{
                .x = 0,
                .y = fb_height,
                .width = fb_width,
                .height = -fb_height,
                .minDepth = 0.0F,
                .maxDepth = 1.0F,
        });
        bound.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        bound.disable_depth();
        bound.enable_alpha_blend();

        auto const left = data.DisplayPos.x;
        auto const right = data.DisplayPos.x + data.DisplaySize.x;
        auto const top = data.DisplayPos.y;
        auto const bottom = data.DisplayPos.y + data.DisplaySize.y;

        // Each command list goes into the rings as it is, with its own vertex address and index buffer, instead of
        // being gathered into one big vector first.
        for (auto const *list: std::span{data.CmdLists.Data, static_cast<std::size_t>(data.CmdListsCount)}) {
            auto const vertex_slice = vertices.write(
                    gpu, frame, std::span<ImDrawVert const>{list->VtxBuffer.Data, static_cast<std::size_t>(list->VtxBuffer.Size)});
            auto const index_slice = indices.write(
                    gpu, frame, std::span<ImDrawIdx const>{list->IdxBuffer.Data, static_cast<std::size_t>(list->IdxBuffer.Size)}, 4);

            if (!vertex_slice || !index_slice) {
                error("[ImGui] Could not write the draw lists: {}", (!vertex_slice ? vertex_slice : index_slice).error().message);
                return;
            }

            bound.bind_index_buffer(*index_slice, VK_INDEX_TYPE_UINT16);

            for (auto const &imgui_cmd: std::span{list->CmdBuffer.Data, static_cast<std::size_t>(list->CmdBuffer.Size)}) {
                auto clip_min = ImVec2{(imgui_cmd.ClipRect.x - data.DisplayPos.x) * data.FramebufferScale.x,
                                       (imgui_cmd.ClipRect.y - data.DisplayPos.y) * data.FramebufferScale.y};
                auto clip_max = ImVec2{(imgui_cmd.ClipRect.z - data.DisplayPos.x) * data.FramebufferScale.x,
                                       (imgui_cmd.ClipRect.w - data.DisplayPos.y) * data.FramebufferScale.y};

                clip_min.x = std::max(clip_min.x, 0.0F);
                clip_min.y = std::max(clip_min.y, 0.0F);
                clip_max.x = std::min(clip_max.x, fb_width);
                clip_max.y = std::min(clip_max.y, fb_height);

                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) {
                    continue;
                }

                auto const raw_tex_id = static_cast<std::uint64_t>(imgui_cmd.GetTexID());
                auto const is_preview = (raw_tex_id & preview_texture_bit) != 0;
                auto const already_linear = is_preview || (raw_tex_id & linear_source_texture_bit) != 0;

                bound.push(PC{
                        .lrtb = {left, right, top, bottom},
                        .vb = vertex_slice->address,
                        .base_vertex = imgui_cmd.VtxOffset,
                        .texture_id = is_preview ? renderer.preview_texture_index()
                                                 : static_cast<std::uint32_t>(raw_tex_id & 0xFFFFFFFFULL),
                        .sampler_id = sampler.index,
                        .already_linear = already_linear ? 1U : 0U,
                        .sdf_text = (raw_tex_id & font_texture_bit) != 0 ? 1U : 0U,
                });

                bound.set_scissor(VkRect2D{
                        .offset = {static_cast<std::int32_t>(clip_min.x), static_cast<std::int32_t>(clip_min.y)},
                        .extent = {static_cast<std::uint32_t>(clip_max.x - clip_min.x),
                                   static_cast<std::uint32_t>(clip_max.y - clip_min.y)},
                });
                bound.draw_indexed(imgui_cmd.ElemCount, 1, imgui_cmd.IdxOffset);
            }
        }

        for (auto *ring: {&vertices, &indices}) {
            if (auto const flushed = ring->flush(frame); !flushed) {
                error("[ImGui] Could not flush an upload buffer: {}", flushed.error().message);
            }
        }
    }

    auto ImGuiRenderer::set_app_name(const std::string_view name) -> void {
        config_name = state_path(std::format("{}.ini", name)).absolute().string();
        config_path = std::make_unique<std::filesystem::path>(config_name);
    }

    auto ImGuiRenderer::update_font(FontChoice const &f) -> void {
        ImGuiIO &io = ImGui::GetIO();
        ImFontConfig cfg{};
        cfg.FontDataOwnedByAtlas = false;
        cfg.SizePixels = std::ceil(f.size);
        cfg.PixelSnapH = true;
        cfg.FontLoader = sdf_font_loader();

        ImFont *font = nullptr;
        auto loaded_font = read_resource(data_path(f.font_path));

        if (loaded_font) {
            font_data = std::move(*loaded_font);
            font = io.Fonts->AddFontFromMemoryTTF(font_data.data(), static_cast<int>(font_data.size()), cfg.SizePixels,
                                                  &cfg);

            // Emoji, as outlines in the text's colour: the same font the world text falls back to.
            if (auto loaded_emoji = read_resource(data_path("assets/fonts/NotoEmoji-Regular.ttf"))) {
                emoji_font_data = std::move(*loaded_emoji);
                // Symbols and arrows, the joiners and variation selectors that ride along with emoji, and the emoji.
                static constexpr std::array<ImWchar, 9> emoji_ranges{0x200B,  0x200D,  0x2190,  0x2BFF, 0xFE00,
                                                                     0xFE0F, 0x1F000, 0x1FAFF, 0};
                ImFontConfig emoji_cfg = cfg;
                emoji_cfg.MergeMode = true;
                static_cast<void>(io.Fonts->AddFontFromMemoryTTF(emoji_font_data.data(),
                                                                 static_cast<int>(emoji_font_data.size()),
                                                                 cfg.SizePixels, &emoji_cfg, emoji_ranges.data()));
            }

            static constexpr std::array<ImWchar, 3> icon_ranges{ICON_MIN_FA, ICON_MAX_16_FA, 0};
            ImFontConfig icon_cfg{};
            icon_cfg.MergeMode = true;
            icon_cfg.PixelSnapH = true;
            icon_cfg.FontLoader = sdf_font_loader();
            icon_cfg.GlyphMinAdvanceX = cfg.SizePixels * 2.0F / 3.0F;
            static_cast<void>(io.Fonts->AddFontFromMemoryCompressedTTF(
                    fa_solid_900_compressed_data, static_cast<int>(fa_solid_900_compressed_size),
                    cfg.SizePixels * 2.0F / 3.0F, &icon_cfg, icon_ranges.data()));
        }

        io.Fonts->Flags |= ImFontAtlasFlags_NoPowerOfTwoHeight;

        io.FontDefault = font;

        auto created_sampler = renderer.sampler_storage().create_sampler(SamplerCreateInfo{
                .mag_filter = VK_FILTER_LINEAR,
                .min_filter = VK_FILTER_LINEAR,
                .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .compare_op = VK_COMPARE_OP_ALWAYS,
                .max_lod = VK_LOD_CLAMP_NONE,
                .sampler_class = SamplerClass::regular,
                .debug_name = "imgui_font_sampler",
        });

        if (!created_sampler) {
            error("[ImGui] Failed to create font sampler");
            return;
        }

        sampler = *created_sampler;
    }

    // Creates, refreshes and releases the images behind ImGui's textures (the glyph atlas grows as text with new
    // glyphs or sizes appears). An update uploads the whole atlas into a new image: the atlas is small and changes
    // rarely, and there is no partial upload.
    auto ImGuiRenderer::update_textures(ImDrawData const &data) -> void {
        auto &images = renderer.image_storage();

        for (auto &retired: retired_textures) {
            if (--retired.frames == 0) {
                static_cast<void>(images.destroy_image(retired.image));
            }
        }
        std::erase_if(retired_textures, [](RetiredTexture const &retired) { return retired.frames == 0; });

        if (data.Textures == nullptr) {
            return;
        }

        for (auto *const texture: *data.Textures) {
            if (texture->Status == ImTextureStatus_OK) {
                continue;
            }

            auto const existing = textures.find(texture);

            if (texture->Status == ImTextureStatus_WantDestroy && static_cast<std::uint32_t>(texture->UnusedFrames) >= retired_texture_frames) {
                if (existing != textures.end()) {
                    retired_textures.push_back({existing->second, retired_texture_frames});
                    textures.erase(existing);
                }
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
                continue;
            }

            if (texture->Status != ImTextureStatus_WantCreate && texture->Status != ImTextureStatus_WantUpdates) {
                continue;
            }

            auto const bytes_per_pixel = static_cast<std::size_t>(texture->BytesPerPixel);
            auto created = images.create_image(
                    ImageCreateInfo{
                            .extent = {static_cast<std::uint32_t>(texture->Width),
                                       static_cast<std::uint32_t>(texture->Height), 1},
                            .format = texture->Format == ImTextureFormat_Alpha8 ? VK_FORMAT_R8_UNORM
                                                                                 : VK_FORMAT_R8G8B8A8_UNORM,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                            .view_type = VK_IMAGE_VIEW_TYPE_2D,
                            .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                            .mip_levels = 1,
                            .array_layers = 1,
                            .debug_name = "imgui_fonts",
                    },
                    std::span<std::byte const>(std::bit_cast<std::byte const *>(texture->GetPixels()),
                                               static_cast<std::size_t>(texture->Width) *
                                                       static_cast<std::size_t>(texture->Height) * bytes_per_pixel));

            if (!created) {
                error("[ImGui] Failed to upload a texture: {}", describe(created.error()));
                continue;
            }

            if (existing != textures.end()) {
                retired_textures.push_back({existing->second, retired_texture_frames});
                existing->second = *created;
            } else {
                textures.emplace(texture, *created);
            }

            texture->SetTexID(ImTextureID{static_cast<std::uint64_t>(created->index) | font_texture_bit});
            texture->SetStatus(ImTextureStatus_OK);
        }
    }

}
