#include "engine_models.hxx"

#include "primitive_meshes.hxx"

#include "renderer.hxx"

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

} // namespace

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

    return EngineModels{
            .cube = *cube,
            .sphere = *sphere,
            .grass_clump = *grass_clump,
            .capsule = *capsule,
    };
}
