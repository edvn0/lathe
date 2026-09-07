#include "rendering/editor_icons.hxx"

#include <array>
#include <format>
#include <string_view>

#include "core/logger.hxx"
#include "rendering/renderer.hxx"

namespace gui {

    namespace {
        struct IconFile {
            EditorIcon icon;
            std::string_view name;
        };

        constexpr std::array icon_files{
                IconFile{EditorIcon::mesh, "mesh"},
                IconFile{EditorIcon::point_light, "point_light"},
                IconFile{EditorIcon::spot_light, "spot_light"},
                IconFile{EditorIcon::script, "script"},
                IconFile{EditorIcon::player, "player"},
                IconFile{EditorIcon::bullet, "bullet"},
                IconFile{EditorIcon::empty, "empty"},
                IconFile{EditorIcon::folder, "folder"},
                IconFile{EditorIcon::search, "search"},
                IconFile{EditorIcon::move, "move"},
                IconFile{EditorIcon::rotate, "rotate"},
                IconFile{EditorIcon::scale, "scale"},
                IconFile{EditorIcon::local, "local"},
                IconFile{EditorIcon::world, "world"},
        };
    } // namespace

    EditorIcons::EditorIcons(Renderer &renderer) {
        for (auto const &file: icon_files) {
            auto const path = std::format("assets/editor/icons/{}.png", file.name);
            auto decoded = DecodedImage::load_from_file(path);

            if (!decoded) {
                error("[EditorIcons] Failed to load '{}' -- falling back to a blank icon", path);
                textures_[static_cast<std::size_t>(file.icon)] = renderer.image_storage().white();
                continue;
            }

            auto created = renderer.image_storage().create_image(
                    ImageCreateInfo{
                            .extent = VkExtent3D{.width = decoded->width(), .height = decoded->height(), .depth = 1},
                            .format = VK_FORMAT_R8G8B8A8_UNORM,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                            .view_type = VK_IMAGE_VIEW_TYPE_2D,
                            .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                            .mip_levels = 1,
                            .array_layers = 1,
                            .debug_name = std::format("editor_icon.{}", file.name),
                    },
                    decoded->span());

            if (!created) {
                error("[EditorIcons] Failed to create GPU image for '{}' -- falling back to a blank icon", path);
                textures_[static_cast<std::size_t>(file.icon)] = renderer.image_storage().white();
                continue;
            }

            textures_[static_cast<std::size_t>(file.icon)] = *created;
        }
    }

    auto EditorIcons::texture(EditorIcon icon) const noexcept -> ImTextureID {
        return ImTextureID{textures_[static_cast<std::size_t>(icon)].index};
    }

} // namespace gui
