#include "rendering/ray_tracing_poc.hxx"

#include <tracy/Tracy.hpp>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

namespace {
    constexpr auto group_size = std::uint32_t{8};

    // Mirrors PushConstants in rt_poc.slang.
    struct Params {
        glm::mat4 inverse_view_projection{1.0F};
        VkDeviceAddress instances = 0;
        ImageRead hdr;
        ImageRead depth;
        ImageWrite result;
        float sun_x = 0.0F;
        float sun_y = 0.0F;
        float sun_z = 0.0F;
        std::uint32_t mode = 0;

        using Layout = PushLayout<&Params::inverse_view_projection, &Params::instances, &Params::hdr, &Params::depth,
                                  &Params::result, &Params::sun_x, &Params::sun_y, &Params::sun_z, &Params::mode>;
    };

    static_assert(GameGraph::push_size<Params>(Params::Layout{}) == 100);
}

auto RayTracingPoc::create(Renderer &renderer) -> std::expected<void, RendererError> {
    if (!renderer.context().ray_query_supported || !renderer.scene_acceleration().ready()) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    auto shader = renderer.game_gpu().register_compute({.source = "assets/shaders/rt_poc.slang",
                                                        .entry_point = "main_cs",
                                                        .debug_name = "rt_poc"});
    if (!shader) {
        return std::unexpected(shader.error());
    }
    shader_ = *shader;
    renderer_ = &renderer;

    renderer.set_ray_tracing_scene_enabled(true);

    ready_ = true;
    info("Ray tracing checks ready (scene TLAS, view '{}')", view_names[view_]);
    return {};
}

auto RayTracingPoc::declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection,
                            glm::vec3 sun_direction) -> void {
    if (!ready_ || graph.slot() != GameSlot::after_lighting) {
        return;
    }
    auto const hdr = graph.scene_hdr();
    auto const depth = graph.scene_depth();
    auto const table = renderer_->scene_acceleration().instance_table_address();
    if (!hdr || !depth || table == 0 || !renderer_->scene_acceleration().active()) {
        return;
    }
    ZoneScopedNC("Ray tracing checks declare", tracy::Color::DeepPink);

    auto const sun = glm::normalize(sun_direction);
    auto params = Params{
            .inverse_view_projection = glm::inverse(projection * view),
            .instances = table,
            .hdr = ImageRead{*hdr},
            .depth = ImageRead{*depth},
            .result = ImageWrite{GameImageDesc{.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "rt_poc"}},
            .sun_x = sun.x,
            .sun_y = sun.y,
            .sun_z = sun.z,
            .mode = view_,
    };

    // On the graphics queue, in order after the scene acceleration structure build, whose closing barrier is what makes
    // the TLAS visible to this pass (a game pass cannot declare a read of an engine resource).
    auto const extent = graph.scene_extent();
    graph.add_compute("rt_poc", params, shader_, Threads::of_extent(extent, group_size, group_size),
                      {.queue = frame_graph::QueueAffinity::graphics, .label = "Ray tracing checks"});
    graph.replace_scene_colour(params.result.image);
}
