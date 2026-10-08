#include "rendering/engine_models.hxx"

#include <format>
#include <string>

#include "assets/primitive_meshes.hxx"
#include "assets/procedural_textures.hxx"

#include "rendering/renderer.hxx"

namespace {

    auto to_renderer_error(ModelLoadError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::model_load_error,
                .cause = ErrorCause{Boxed<ModelLoadError>{std::move(error)}},
        };
    }

    auto create_primitive_model(Renderer &renderer, std::expected<PrimitiveMeshData, ModelLoadError> mesh)
            -> std::expected<ModelHandle, RendererError> {
        if (!mesh) {
            return std::unexpected(to_renderer_error(mesh.error()));
        }

        return renderer.create_model_from_cpu_data(to_model_cpu_data(std::move(*mesh)));
    }

    constexpr float grass_card_alpha_cutoff = 0.5F;

    auto create_grass_card_texture(Renderer &renderer) -> std::expected<ImageHandle, RendererError> {
        auto const texture = make_grass_card_texture(128, grass_card_alpha_cutoff);

        auto image = renderer.image_storage().create_image(
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = texture.width, .height = texture.height, .depth = 1},
                        .format = VK_FORMAT_R8G8B8A8_SRGB,
                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .mip_levels = texture.mip_levels,
                        .array_layers = 1,
                        .debug_name = "engine.grass_card_texture",
                },
                texture.pixels, ImageMipSource::provided);

        if (!image) {
            return std::unexpected(RendererError{
                    .type = RendererErrorType::image_error,
                    .cause = ErrorCause{Boxed<ImageStorageError>{image.error()}},
            });
        }

        return *image;
    }

    auto create_or_update_material(Renderer &renderer, MaterialHandle handle, MaterialCreateInfo const &info,
                                   std::string name) -> std::expected<MaterialHandle, RendererError> {
        if (handle.valid()) {
            if (auto updated = renderer.update_material(handle, info); !updated) {
                return std::unexpected(updated.error());
            }
            return handle;
        }
        return renderer.create_material(info, std::move(name));
    }

}

auto create_engine_models(Renderer &renderer) -> std::expected<EngineModels, RendererError> {
    auto cube = create_primitive_model(renderer, make_cube_mesh());

    if (!cube) {
        return std::unexpected(cube.error());
    }

    auto sphere = create_primitive_model(renderer, make_sphere_mesh());

    if (!sphere) {
        return std::unexpected(sphere.error());
    }

    auto grass_clump = create_primitive_model(renderer, make_grass_clump_mesh());

    if (!grass_clump) {
        return std::unexpected(grass_clump.error());
    }

    auto capsule = create_primitive_model(renderer, make_capsule_mesh());

    if (!capsule) {
        return std::unexpected(capsule.error());
    }

    auto grass_card_texture = create_grass_card_texture(renderer);

    if (!grass_card_texture) {
        return std::unexpected(grass_card_texture.error());
    }

    renderer.register_model_name(*cube, "Cube");
    renderer.register_model_name(*sphere, "Sphere");
    renderer.register_model_name(*grass_clump, "Grass Clump");
    renderer.register_model_name(*capsule, "Capsule");

    return EngineModels{
            .cube = *cube,
            .sphere = *sphere,
            .grass_clump = *grass_clump,
            .capsule = *capsule,
            .grass_card_texture = *grass_card_texture,
    };
}

auto grass_materials(Renderer &renderer, EngineModels const &engine_models, MaterialCreateInfo const &look,
                     GrassMaterials existing, std::string_view name) -> std::expected<GrassMaterials, RendererError> {
    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    auto cards_info = look;
    cards_info.base_colour_texture = engine_models.grass_card_texture;
    cards_info.sampler = samplers.linear_clamp();
    cards_info.alpha_mode = AlphaMode::mask;
    cards_info.alpha_cutoff = grass_card_alpha_cutoff;
    cards_info.alpha_to_coverage = true;
    cards_info.double_sided = true;
    cards_info.far_material = {};

    auto cards = create_or_update_material(renderer, existing.cards, cards_info,
                                           name.empty() ? std::string{} : std::format("{}_cards", name));
    if (!cards) {
        return std::unexpected(cards.error());
    }

    auto blades_info = look;
    if (!blades_info.base_colour_texture.valid()) {
        blades_info.base_colour_texture = images.white();
    }
    blades_info.alpha_mode = AlphaMode::opaque;
    blades_info.alpha_to_coverage = false;
    blades_info.double_sided = true;
    blades_info.far_material = *cards;
    blades_info.far_material_lod = grass_clump_card_lod;

    auto blades = create_or_update_material(renderer, existing.blades, blades_info, std::string{name});
    if (!blades) {
        if (!existing.cards.valid()) {
            renderer.release_material(*cards);
        }
        return std::unexpected(blades.error());
    }

    return GrassMaterials{.blades = *blades, .cards = *cards};
}
