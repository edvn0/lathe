#include "rendering/text_renderer.hxx"

#include <algorithm>
#include <bit>
#include <format>

#include <entt/entt.hpp>
#include <tracy/Tracy.hpp>

#include "core/logger.hxx"
#include "core/paths.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/text_gpu.hxx"

namespace {
    constexpr auto group_size = std::uint32_t{64};
    constexpr auto text_colour = std::uint32_t{0x8A2BE2};

    // The members in Layout, in order, are the push constants of the shader with the same name: text_<pass>.slang.
    // Buffers push their address (8 bytes), images their bindless index (4), and the plain VkDeviceAddress members
    // are the addresses of this frame's uploads, which the CPU wrote and no pass needs to declare.
    struct RasterParams {
        VkDeviceAddress segments = 0;
        VkDeviceAddress jobs = 0;
        BufferWrite tiles;
        std::uint32_t job_count = 0;
        std::uint32_t texel_count = 0;

        using Layout = PushLayout<&RasterParams::segments, &RasterParams::jobs, &RasterParams::tiles,
                                  &RasterParams::job_count, &RasterParams::texel_count>;
    };

    struct PlaceParams {
        VkDeviceAddress jobs = 0;
        BufferReadWrite rects;
        BufferReadWrite state;
        std::uint32_t job_count = 0;
        std::uint32_t reset = 0;

        using Layout = PushLayout<&PlaceParams::jobs, &PlaceParams::rects, &PlaceParams::state, &PlaceParams::job_count,
                                  &PlaceParams::reset>;
    };

    struct BlitParams {
        BufferRead tiles;
        VkDeviceAddress jobs = 0;
        BufferRead rects;
        BufferReadWrite atlas;
        std::uint32_t job_count = 0;
        std::uint32_t quad_count = 0;

        using Layout = PushLayout<&BlitParams::tiles, &BlitParams::jobs, &BlitParams::rects, &BlitParams::atlas,
                                  &BlitParams::job_count, &BlitParams::quad_count>;
    };

    struct BinParams {
        VkDeviceAddress instances = 0;
        BufferReadWrite counts;
        BufferWrite lists;
        std::uint32_t instance_count = 0;
        std::uint32_t tiles_x = 0;
        std::uint32_t tiles_y = 0;

        using Layout = PushLayout<&BinParams::instances, &BinParams::counts, &BinParams::lists,
                                  &BinParams::instance_count, &BinParams::tiles_x, &BinParams::tiles_y>;
    };

    struct CompositeParams {
        ImageRead scene;
        ImageWrite result;
        BufferRead atlas;
        BufferRead rects;
        BufferRead counts;
        BufferRead lists;
        VkDeviceAddress instances = 0;
        std::uint32_t tiles_x = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;

        using Layout = PushLayout<&CompositeParams::scene, &CompositeParams::result, &CompositeParams::atlas,
                                  &CompositeParams::rects, &CompositeParams::counts, &CompositeParams::lists,
                                  &CompositeParams::instances, &CompositeParams::tiles_x, &CompositeParams::width,
                                  &CompositeParams::height>;
    };

    // Mirrors WorldPushConstants in text_world.slang.
    struct WorldPush {
        glm::mat4 view_projection{1.0F};
        VkDeviceAddress atlas = 0;
        VkDeviceAddress rects = 0;
        VkDeviceAddress glyphs = 0;
        VkDeviceAddress transforms = 0;
    };

    static_assert(GameGraph::push_size<RasterParams>(RasterParams::Layout{}) == 32);
    static_assert(GameGraph::push_size<PlaceParams>(PlaceParams::Layout{}) == 32);
    static_assert(GameGraph::push_size<BlitParams>(BlitParams::Layout{}) == 40);
    static_assert(GameGraph::push_size<BinParams>(BinParams::Layout{}) == 36);
    static_assert(GameGraph::push_size<CompositeParams>(CompositeParams::Layout{}) == 60);
    static_assert(sizeof(WorldPush) == 96);

    constexpr auto atlas_bytes = VkDeviceSize{text::atlas_size} * text::atlas_size;
    constexpr auto rect_bytes = VkDeviceSize{text::max_slots} * sizeof(text::AtlasRect);
    // The shelf's cursor and height: three words, padded to a vec4.
    constexpr auto state_bytes = VkDeviceSize{16};

    auto ceil_div(std::uint32_t value, std::uint32_t divisor) noexcept -> std::uint32_t {
        return (value + divisor - 1) / divisor;
    }
} // namespace

auto TextRenderer::create(Renderer &renderer, GameGpu &gpu) -> std::expected<void, RendererError> {
    ZoneScopedNC("Text renderer create", tracy::Color::BlueViolet);

    if (ready_) {
        return {};
    }

    auto const compute = [&](char const *name) {
        return gpu.register_compute({
                .source = std::format("assets/shaders/{}.slang", name),
                .debug_name = name,
        });
    };

    auto raster = compute("text_raster");
    auto place = compute("text_place");
    auto blit = compute("text_blit");
    auto bin = compute("text_bin");
    auto composite = compute("text_composite");
    for (auto const *registered: {&raster, &place, &blit, &bin, &composite}) {
        if (!*registered) {
            return std::unexpected(registered->error());
        }
    }

    auto world = gpu.register_graphics({
            .source = "assets/shaders/text_world.slang",
            .blending = true,
            .debug_name = "text_world",
    });
    if (!world) {
        return std::unexpected(world.error());
    }

    renderer_ = &renderer;
    gpu_ = &gpu;
    raster_ = *raster;
    place_ = *place;
    blit_ = *blit;
    bin_ = *bin;
    composite_ = *composite;
    world_ = *world;

    if (auto const font = load_font(data_path("assets/fonts/GoogleSansCode-Regular.ttf").absolute()); !font) {
        warn("[Text] The built-in font could not be loaded: {}", font.error());
    }

    ready_ = true;
    return {};
}

auto TextRenderer::load_font(std::filesystem::path const &path) -> std::expected<text::FontId, std::string> {
    auto const absolute = path.is_absolute() ? path : data_path(path.generic_string()).absolute();

    auto font = Font::load(absolute);
    if (!font) {
        return std::unexpected(font.error());
    }

    auto const lock = std::scoped_lock{mutex_};
    return cache_.add_font(std::move(*font));
}

auto TextRenderer::draw(text::Draw draw) -> void {
    auto const lock = std::scoped_lock{mutex_};
    pending_.push_back(std::move(draw));
}

auto TextRenderer::draw_world(text::WorldDraw draw) -> void {
    auto const lock = std::scoped_lock{mutex_};
    pending_world_.push_back(std::move(draw));
}

auto TextRenderer::measure(text::FontId font, std::string_view utf8, float size) -> glm::vec2 {
    auto const lock = std::scoped_lock{mutex_};
    return cache_.measure(font, utf8, size);
}

auto TextRenderer::declare(GameGraph &graph, entt::registry const *registry) -> void {
    if (!ready_) {
        return;
    }

    switch (graph.slot()) {
        case GameSlot::frame_start:
            declare_glyphs_and_world(graph, registry);
            break;
        case GameSlot::before_composite:
            declare_screen(graph);
            break;
        default:
            break;
    }
}

auto TextRenderer::declare_glyphs_and_world(GameGraph &graph, entt::registry const *registry) -> void {
    frame_ = {};
    stats_ = {};

    // Open the frame's upload buffers even when there is nothing to draw: a particle emitter may still ask for
    // outline points after this.
    auto &device = renderer_->context();
    auto const slot = graph.frame_slot();

    upload_open_ = false;
    if (auto const begun = upload_.begin_frame(device, slot); !begun) {
        error("[Text] Could not prepare the upload buffer: {}", begun.error().message);
    } else {
        upload_open_ = true;
        upload_slot_ = slot;
    }

    auto draws = std::vector<text::Draw>{};
    auto world_draws = std::vector<text::WorldDraw>{};
    auto batch = text::Batch{};

    {
        auto const lock = std::scoped_lock{mutex_};
        draws.swap(pending_);
        world_draws.swap(pending_world_);

        if (registry != nullptr) {
            for (auto const entity: registry->view<Components::Text const>()) {
                auto const &entry = registry->get<Components::Text const>(entity);
                if (!entry.visible || entry.text.empty()) {
                    continue;
                }

                auto const *const transform = registry->try_get<Components::Transform>(entity);
                world_draws.push_back(text::WorldDraw{
                        .font = entry.font,
                        .text = entry.text,
                        .transform = transform != nullptr ? systems::get_world_transform(*registry, entity, *transform)
                                                          : glm::mat4{1.0F},
                        .size = entry.size,
                        .colour = entry.colour,
                        .intensity = entry.emissive,
                        .centred = entry.centred,
                });
            }
        }

        if (draws.empty() && world_draws.empty()) {
            return;
        }

        batch = cache_.build(draws, world_draws);
    }

    stats_ = {
            .new_glyphs = static_cast<std::uint32_t>(batch.jobs.size()),
            .screen_glyphs = static_cast<std::uint32_t>(batch.instances.size()),
            .world_glyphs = static_cast<std::uint32_t>(batch.world_instances.size()),
            .atlas_reset = batch.reset_atlas,
    };

    if (batch.empty()) {
        return;
    }
    ZoneScopedNC("Text declare", tracy::Color::BlueViolet);

    if (!upload_open_) {
        return;
    }

    auto failed = false;
    auto const put = [&]<typename T>(std::vector<T> const &items) {
        auto written = upload_.write(device, slot, std::span<T const>{items});
        if (!written) {
            error("[Text] Could not upload this frame's text: {}", written.error().message);
            failed = true;
            return UploadSlice{};
        }
        return *written;
    };

    auto const segments = put(batch.segments);
    auto const jobs = put(batch.jobs);
    auto const instances = put(batch.instances);
    auto const world_glyphs = put(batch.world_instances);
    auto const transforms = put(batch.world_transforms);

    if (auto const flushed = upload_.flush(slot); !flushed) {
        error("[Text] Could not flush this frame's text: {}", flushed.error().message);
        failed = true;
    }
    if (failed) {
        return;
    }

    frame_ = {
            .screen_text = !batch.instances.empty(),
            .instances = instances,
            .instance_count = static_cast<std::uint32_t>(batch.instances.size()),
    };

    graph.isolated("text glyphs", [&] {
        auto atlas = graph.persistent_buffer("text.atlas", {.size = atlas_bytes});
        auto rects = graph.persistent_buffer("text.rects", {.size = rect_bytes});
        auto state = graph.persistent_buffer("text.state", {.size = state_bytes});

        if (!batch.jobs.empty()) {
            auto const job_count = static_cast<std::uint32_t>(batch.jobs.size());

            // One thread per texel of every new tile: outline -> signed distance.
            auto const tiles =
                    graph.create_buffer({.size = VkDeviceSize{batch.texel_count} * sizeof(float)}, "text.tiles");
            auto raster = RasterParams{.segments = segments.address,
                                       .jobs = jobs.address,
                                       .tiles = {tiles},
                                       .job_count = job_count,
                                       .texel_count = batch.texel_count};
            graph.add_compute("text_raster", raster, raster_, Threads{.x = batch.texel_count, .group_x = group_size},
                              {.side_effect = true, .label = "Text raster", .color = text_colour});

            // One thread walks the shelves: which rectangle of the atlas each new tile gets.
            auto place = PlaceParams{.jobs = jobs.address,
                                     .rects = {rects},
                                     .state = {state},
                                     .job_count = job_count,
                                     .reset = batch.reset_atlas ? 1U : 0U};
            graph.add_compute("text_place", place, place_, Threads{.x = 1, .group_x = 1},
                              {.side_effect = true, .label = "Text pack: place", .color = text_colour});
            rects = place.rects.buffer;
            state = place.state.buffer;

            // Four texels per thread: the tiles into their rectangles, as bytes.
            auto blit = BlitParams{.tiles = {raster.tiles.buffer},
                                   .jobs = jobs.address,
                                   .rects = {rects},
                                   .atlas = {atlas},
                                   .job_count = job_count,
                                   .quad_count = batch.texel_count / 4U};
            graph.add_compute("text_blit", blit, blit_, Threads{.x = batch.texel_count / 4U, .group_x = group_size},
                              {.side_effect = true, .label = "Text pack: blit", .color = text_colour});
            atlas = blit.atlas.buffer;
        }

        if (!batch.world_instances.empty()) {
            // The draw reads what the packing wrote, so the forward pass is ordered after it.
            graph.add_scene_draw("text_world", [&](GameDrawBuilder &draw) {
                auto const stages = frame_graph::stages_of(frame_graph::ShaderStage::vertex) |
                                    frame_graph::stages_of(frame_graph::ShaderStage::fragment);
                auto const atlas_read = draw.read(atlas, stages);
                auto const rects_read = draw.read(rects, stages);

                return [this, atlas_read, rects_read, glyphs = world_glyphs.address, transforms = transforms.address,
                        count = static_cast<std::uint32_t>(batch.world_instances.size())](GameDrawContext &context) {
                    context.bind(world_);
                    context.push(WorldPush{
                            .view_projection = context.view_projection(),
                            .atlas = context.address(atlas_read),
                            .rects = context.address(rects_read),
                            .glyphs = glyphs,
                            .transforms = transforms,
                    });
                    context.draw(count * 6U);
                };
            });
        }
    });
}

auto TextRenderer::declare_screen(GameGraph &graph) -> void {
    if (!frame_.screen_text) {
        return;
    }
    ZoneScopedNC("Text screen declare", tracy::Color::BlueViolet);

    auto const scene = graph.scene_colour_source();
    auto const extent = graph.scene_extent();
    if (!scene || extent.width == 0 || extent.height == 0) {
        return;
    }

    graph.isolated("text screen", [&] {
        auto const tiles_x = ceil_div(extent.width, text::screen_tile);
        auto const tiles_y = ceil_div(extent.height, text::screen_tile);
        auto const tile_count = VkDeviceSize{tiles_x} * tiles_y;

        auto const atlas = graph.persistent_buffer("text.atlas", {.size = atlas_bytes});
        auto const rects = graph.persistent_buffer("text.rects", {.size = rect_bytes});
        auto const counts =
                graph.create_buffer({.size = tile_count * sizeof(std::uint32_t), .zero = true}, "text.counts");
        auto const lists =
                graph.create_buffer({.size = tile_count * text::tile_capacity * sizeof(std::uint32_t)}, "text.lists");

        // One thread per glyph quad: which screen tiles it touches.
        auto bin = BinParams{.instances = frame_.instances.address,
                             .counts = {counts},
                             .lists = {lists},
                             .instance_count = frame_.instance_count,
                             .tiles_x = tiles_x,
                             .tiles_y = tiles_y};
        graph.add_compute("text_bin", bin, bin_, Threads{.x = frame_.instance_count, .group_x = group_size},
                          {.label = "Text bin", .color = text_colour});

        // One 16x16 group per screen tile: sort its glyphs into draw order and blend them over the scene.
        auto composite = CompositeParams{
                .scene = *scene,
                .result = ImageWrite{GameImageDesc{.format = renderer_->hdr_format(), .name = "text.scene"}},
                .atlas = {atlas},
                .rects = {rects},
                .counts = {bin.counts.buffer},
                .lists = {bin.lists.buffer},
                .instances = frame_.instances.address,
                .tiles_x = tiles_x,
                .width = extent.width,
                .height = extent.height,
        };
        graph.add_compute("text_composite", composite, composite_,
                          Threads::of_extent(extent, text::screen_tile, text::screen_tile),
                          {.label = "Text composite", .color = text_colour});
        graph.replace_scene_colour(composite.result.image);
    });
}

auto TextRenderer::emission_points(Components::Text const &text) -> std::optional<EmissionPoints> {
    if (!ready_ || !upload_open_ || text.text.empty() || !(text.size > 0.0F)) {
        return std::nullopt;
    }

    auto const key = std::format("{}|{}|{:08x}|{}", text.font, text.centred ? 1 : 0,
                                 std::bit_cast<std::uint32_t>(text.size), text.text);

    auto found = points_.find(key);
    if (found == points_.end()) {
        auto points = std::vector<glm::vec2>{};
        {
            auto const lock = std::scoped_lock{mutex_};
            points = cache_.outline_points(text.font, text.text, text.centred, emission_point_count);
        }
        for (auto &point: points) {
            point *= text.size;
        }

        // Only the render thread comes here, so this is the one place the map grows.
        if (points_.size() >= 64) {
            points_.clear();
        }
        found = points_.emplace(key, std::move(points)).first;
    }

    if (found->second.empty()) {
        return std::nullopt;
    }

    auto &device = renderer_->context();
    auto const written = upload_.write(device, upload_slot_, std::span<glm::vec2 const>{found->second}, 8);
    if (!written) {
        error("[Text] Could not upload emission points: {}", written.error().message);
        return std::nullopt;
    }
    if (auto const flushed = upload_.flush(upload_slot_); !flushed) {
        error("[Text] Could not flush emission points: {}", flushed.error().message);
        return std::nullopt;
    }

    return EmissionPoints{.address = written->address, .count = static_cast<std::uint32_t>(found->second.size())};
}
