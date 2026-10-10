#include "rendering/effect_system.hxx"

#include <algorithm>
#include <format>
#include <fstream>
#include <sstream>
#include <utility>

#include <tracy/Tracy.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"

namespace {
    constexpr auto manifest_poll_frames = std::uint32_t{30};
    constexpr auto effect_colour = std::uint32_t{0x9ACD32};

    auto read_text(std::filesystem::path const &path) -> std::optional<std::string> {
        auto file = std::ifstream{path, std::ios::binary};
        if (!file) {
            return std::nullopt;
        }
        auto contents = std::ostringstream{};
        contents << file.rdbuf();
        return std::move(contents).str();
    }

    auto source_name(EffectSource source) -> std::string_view {
        return source == EffectSource::scene_colour ? "scene_colour" : "scene_depth";
    }

    auto names_of(EffectManifest const &manifest) -> std::string {
        auto names = std::string{};
        for (auto const &binding: manifest.bindings) {
            names += (names.empty() ? "" : ", ") + binding.name;
        }
        for (auto const &param: manifest.params) {
            names += (names.empty() ? "" : ", ") + param.name;
        }
        return names;
    }

    auto is_path_allowed(std::string_view path) -> bool {
        return path.starts_with("assets/") && path.ends_with(".json") && path.find("..") == std::string_view::npos &&
               path.find('\\') == std::string_view::npos;
    }
}

auto EffectSystem::create(GameGpu &gpu) -> void {
    gpu_ = &gpu;
    registrar_ = [&gpu](EffectManifest const &manifest) -> std::expected<GameComputeShader, std::string> {
        auto registered = gpu.register_compute({
                .source = "assets/shaders/" + manifest.shader,
                .entry_point = manifest.entry,
                .debug_name = "effect." + manifest.shader,
        });
        if (!registered) {
            return std::unexpected(describe(registered.error()));
        }
        return *registered;
    };
}

auto EffectSystem::define(std::string name, std::string_view manifest_json)
        -> std::expected<EffectShaderId, std::string> {
    if (!registrar_) {
        return std::unexpected("effects are not available");
    }
    auto manifest = parse_effect_manifest(manifest_json);
    if (!manifest) {
        return std::unexpected(std::format("effect '{}': {}", name, manifest.error()));
    }
    auto compute = registrar_(*manifest);
    if (!compute) {
        return std::unexpected(std::format("effect '{}': could not register '{}': {}", name, manifest->shader,
                                           compute.error()));
    }

    shaders_.push_back(Shader{.name = std::move(name), .manifest = std::move(*manifest), .compute = *compute});
    return static_cast<EffectShaderId>(shaders_.size() - 1);
}

auto EffectSystem::load(std::string_view path) -> std::expected<EffectShaderId, std::string> {
    if (!is_path_allowed(path)) {
        return std::unexpected(std::format("'{}' is not an effect manifest: it must be a .json file under assets/", path));
    }
    auto const data = Paths::current().data(path);
    if (!data) {
        return std::unexpected(std::format("'{}' is not a valid data path", path));
    }

    return load_file(data->absolute(), std::string{path});
}

auto EffectSystem::load_file(std::filesystem::path const &file, std::string name)
        -> std::expected<EffectShaderId, std::string> {
    for (auto index = std::size_t{0}; index < shaders_.size(); ++index) {
        if (shaders_[index].file == file) {
            return static_cast<EffectShaderId>(index);
        }
    }

    auto text = read_text(file);
    if (!text) {
        return std::unexpected(std::format("could not read '{}'", name));
    }
    auto defined = define(std::move(name), *text);
    if (!defined) {
        return defined;
    }

    auto &shader = shaders_[*defined];
    shader.file = file;
    std::error_code ignored;
    shader.modified = std::filesystem::last_write_time(file, ignored);
    return defined;
}

auto EffectSystem::manifest(EffectShaderId shader) const noexcept -> EffectManifest const * {
    return shader < shaders_.size() ? &shaders_[shader].manifest : nullptr;
}

auto EffectSystem::reload(EffectShaderId id) -> std::expected<void, std::string> {
    if (id >= shaders_.size() || !registrar_) {
        return std::unexpected("no such effect shader");
    }
    auto &shader = shaders_[id];
    if (shader.file.empty()) {
        return std::unexpected(std::format("effect '{}' was not loaded from a file", shader.name));
    }

    auto text = read_text(shader.file);
    if (!text) {
        return std::unexpected(std::format("could not read '{}'", shader.name));
    }
    auto manifest = parse_effect_manifest(*text);
    if (!manifest) {
        return std::unexpected(std::format("effect '{}': {}", shader.name, manifest.error()));
    }

    // A new source or entry point is a new shader object; an edit to the same file is the watcher's business.
    auto compute = shader.compute;
    if (manifest->shader != shader.manifest.shader || manifest->entry != shader.manifest.entry) {
        auto registered = registrar_(*manifest);
        if (!registered) {
            return std::unexpected(std::format("effect '{}': could not register '{}': {}", shader.name,
                                               manifest->shader, registered.error()));
        }
        compute = *registered;
    }

    // Instances keep what still fits, and are re-checked against their slot; one that no longer fits is taken off.
    for (auto &[effect_id, effect]: effects_) {
        if (effect.shader != id) {
            continue;
        }
        effect = fit(effect, shader.manifest, *manifest);
    }
    shader.manifest = std::move(*manifest);
    shader.compute = compute;
    std::error_code ignored;
    shader.modified = std::filesystem::last_write_time(shader.file, ignored);

    for (auto &[effect_id, effect] : effects_) {
        if (effect.shader == id && effect.slot) {
            if (auto const problem = validate(effect, *effect.slot)) {
                effect.problem = std::format("after reloading '{}': {}", shader.name, *problem);
                remove(effect_id);
            }
        }
    }
    return {};
}

auto EffectSystem::poll_manifests() -> void {
    for (auto index = std::size_t{0}; index < shaders_.size(); ++index) {
        auto &shader = shaders_[index];
        if (shader.file.empty()) {
            continue;
        }
        std::error_code ignored;
        auto const modified = std::filesystem::last_write_time(shader.file, ignored);
        if (ignored || modified == shader.modified) {
            continue;
        }
        shader.modified = modified;
        if (auto const reloaded = reload(static_cast<EffectShaderId>(index)); !reloaded) {
            error("Effect manifest not reloaded, keeping the old one: {}", reloaded.error());
        } else {
            info("Reloaded effect manifest '{}'", shader.name);
        }
    }
}

auto EffectSystem::fit(Effect const &old_effect, EffectManifest const &old_manifest, EffectManifest const &manifest)
        -> Effect {
    auto fitted = Effect{.shader = old_effect.shader, .slot = old_effect.slot, .problem = old_effect.problem};

    for (auto const &param: manifest.params) {
        fitted.params.push_back(param.value);
        auto const old = std::ranges::find(old_manifest.params, param.name, &EffectParam::name);
        if (old == old_manifest.params.end() || old->type != param.type) {
            continue;
        }
        auto const index = static_cast<std::size_t>(old - old_manifest.params.begin());
        if (check_effect_value(param, std::span<double const>{old_effect.params[index].data(),
                                                              component_count(param.type)})) {
            fitted.params.back() = old_effect.params[index];
        }
    }

    for (auto const &binding: manifest.bindings) {
        auto source = EffectSource::scene_colour;
        if (auto const *image = std::get_if<EffectImageIn>(&binding.what)) {
            source = image->source;
        }
        fitted.sources.push_back(source);
        fitted.buffers.emplace_back();

        auto const old = std::ranges::find(old_manifest.bindings, binding.name, &EffectBinding::name);
        if (old == old_manifest.bindings.end() || old->what.index() != binding.what.index()) {
            continue;
        }
        auto const index = static_cast<std::size_t>(old - old_manifest.bindings.begin());
        fitted.sources.back() = old_effect.sources[index];
        fitted.buffers.back() = old_effect.buffers[index];
    }

    // A buffer that is no longer bound is let go of.
    for (auto index = std::size_t{0}; index < old_effect.buffers.size(); ++index) {
        if (!old_effect.buffers[index]) {
            continue;
        }
        auto const kept = std::ranges::any_of(fitted.buffers, [&](auto const &buffer) {
            return buffer == old_effect.buffers[index];
        });
        if (!kept) {
            release_buffer(*old_effect.buffers[index]);
        }
    }
    return fitted;
}

auto EffectSystem::instance(EffectShaderId shader) -> std::expected<EffectId, std::string> {
    if (shader >= shaders_.size()) {
        return std::unexpected("no such effect shader");
    }
    auto const &manifest = shaders_[shader].manifest;

    auto effect = Effect{.shader = shader};
    for (auto const &param: manifest.params) {
        effect.params.push_back(param.value);
    }
    for (auto const &binding: manifest.bindings) {
        effect.sources.push_back(std::get_if<EffectImageIn>(&binding.what) != nullptr
                                         ? std::get<EffectImageIn>(binding.what).source
                                         : EffectSource::scene_colour);
        effect.buffers.emplace_back();
    }

    auto const id = next_effect_++;
    effects_.emplace(id, std::move(effect));
    return id;
}

auto EffectSystem::set(EffectId id, std::string_view name, EffectValue const &value)
        -> std::expected<void, std::string> {
    auto const found = effects_.find(id);
    if (found == effects_.end()) {
        return std::unexpected("the effect no longer exists");
    }
    auto &effect = found->second;
    auto const &manifest = shaders_[effect.shader].manifest;

    if (auto const *param = manifest.find_param(name)) {
        if (value.text || value.buffer) {
            return std::unexpected(std::format("'{}' is a number{}", name, component_count(param->type) > 1 ? " list" : ""));
        }
        auto checked = check_effect_value(*param, value.numbers);
        if (!checked) {
            return std::unexpected(std::format("'{}' {}", name, checked.error()));
        }
        effect.params[static_cast<std::size_t>(param - manifest.params.data())] = *checked;
        return {};
    }

    if (auto const *binding = manifest.find_binding(name)) {
        auto const index = static_cast<std::size_t>(binding - manifest.bindings.data());

        if (std::holds_alternative<EffectImageIn>(binding->what)) {
            auto const source = value.text == "scene_colour"  ? std::optional{EffectSource::scene_colour}
                                : value.text == "scene_depth" ? std::optional{EffectSource::scene_depth}
                                                              : std::nullopt;
            if (!source) {
                return std::unexpected(std::format("'{}' must be \"scene_colour\" or \"scene_depth\"", name));
            }
            if (effect.slot && !source_available(*source, *effect.slot)) {
                return std::unexpected(std::format("{} is not available in {}", source_name(*source),
                                                   game_slot_name(*effect.slot)));
            }
            effect.sources[index] = *source;
            return {};
        }

        if (auto const *buffer = std::get_if<EffectBufferBinding>(&binding->what)) {
            if (!value.buffer || !buffers_.contains(*value.buffer)) {
                return std::unexpected(std::format("'{}' must be a buffer from compute.buffer", name));
            }
            if (buffers_.at(*value.buffer).elements < buffer->elements) {
                return std::unexpected(std::format("'{}' needs a buffer of at least {} floats", name, buffer->elements));
            }
            retain_buffer(*value.buffer);
            if (effect.buffers[index]) {
                release_buffer(*effect.buffers[index]);
            }
            effect.buffers[index] = value.buffer;
            return {};
        }

        return std::unexpected(std::format("'{}' is an output the engine creates", name));
    }

    return std::unexpected(std::format("unknown name '{}'; this effect has: {}", name, names_of(manifest)));
}

auto EffectSystem::source_available(EffectSource source, GameSlot slot) noexcept -> bool {
    return source == EffectSource::scene_colour ? slot >= GameSlot::after_lighting : slot >= GameSlot::after_depth;
}

auto EffectSystem::validate(Effect const &effect, GameSlot slot) const -> std::optional<std::string> {
    auto const &manifest = shaders_[effect.shader].manifest;

    for (auto index = std::size_t{0}; index < manifest.bindings.size(); ++index) {
        auto const &binding = manifest.bindings[index];

        if (std::holds_alternative<EffectImageIn>(binding.what) && !source_available(effect.sources[index], slot)) {
            return std::format("'{}' reads {}, which is not available in {}", binding.name,
                               source_name(effect.sources[index]), game_slot_name(slot));
        }
        if (auto const *image = std::get_if<EffectImageOut>(&binding.what);
            image != nullptr && image->replaces_scene_colour && slot < GameSlot::after_lighting) {
            return std::format("'{}' replaces the scene colour, which only exists from after_lighting on",
                               binding.name);
        }
        if (auto const *buffer = std::get_if<EffectBufferBinding>(&binding.what);
            buffer != nullptr && buffer->elements == 0 && !effect.buffers[index]) {
            return std::format("'{}' needs a buffer from compute.buffer", binding.name);
        }
    }
    return std::nullopt;
}

auto EffectSystem::add(EffectId id, GameSlot slot) -> std::expected<void, std::string> {
    auto const found = effects_.find(id);
    if (found == effects_.end()) {
        return std::unexpected("the effect no longer exists");
    }
    auto &effect = found->second;

    if (effect.slot) {
        return std::unexpected(std::format("the effect is already in {}", game_slot_name(*effect.slot)));
    }
    if (auto const problem = validate(effect, slot)) {
        return std::unexpected(*problem);
    }

    effect.slot = slot;
    effect.problem.clear();
    order_.push_back(id);
    return {};
}

auto EffectSystem::remove(EffectId id) -> void {
    if (auto const found = effects_.find(id); found != effects_.end()) {
        found->second.slot.reset();
    }
    std::erase(order_, id);
}

auto EffectSystem::destroy(EffectId id) -> void {
    auto const found = effects_.find(id);
    if (found == effects_.end()) {
        return;
    }
    remove(id);

    auto const &manifest = shaders_[found->second.shader].manifest;
    for (auto index = std::size_t{0}; index < manifest.bindings.size(); ++index) {
        if (found->second.buffers[index]) {
            release_buffer(*found->second.buffers[index]);
        }
        if (auto const *buffer = std::get_if<EffectBufferBinding>(&manifest.bindings[index].what);
            buffer != nullptr && buffer->elements != 0) {
            unused_buffers_.push_back(owned_buffer_name(id, manifest.bindings[index].name));
        }
    }
    effects_.erase(found);
}

auto EffectSystem::problem(EffectId id) const -> std::string {
    auto const found = effects_.find(id);
    return found != effects_.end() ? found->second.problem : std::string{};
}

auto EffectSystem::slot_of(EffectId id) const noexcept -> std::optional<GameSlot> {
    auto const found = effects_.find(id);
    return found != effects_.end() ? found->second.slot : std::nullopt;
}

auto EffectSystem::create_buffer(std::uint32_t elements) -> std::expected<EffectBufferId, std::string> {
    if (elements == 0 || elements > max_effect_buffer_elements) {
        return std::unexpected(std::format("a buffer holds 1 to {} floats", max_effect_buffer_elements));
    }
    auto const id = next_buffer_++;
    buffers_.emplace(id, Buffer{.elements = elements, .references = 1});
    return id;
}

auto EffectSystem::retain_buffer(EffectBufferId id) -> void {
    if (auto const found = buffers_.find(id); found != buffers_.end()) {
        ++found->second.references;
    }
}

auto EffectSystem::release_buffer(EffectBufferId id) -> void {
    auto const found = buffers_.find(id);
    if (found == buffers_.end()) {
        return;
    }
    if (--found->second.references == 0) {
        unused_buffers_.push_back(std::format("effect_buffer.{}", id));
        buffers_.erase(found);
    }
}

auto EffectSystem::buffer_elements(EffectBufferId id) const noexcept -> std::uint32_t {
    auto const found = buffers_.find(id);
    return found != buffers_.end() ? found->second.elements : 0;
}

auto EffectSystem::owned_buffer_name(EffectId id, std::string_view binding) const -> std::string {
    return std::format("effect.{}.{}", id, binding);
}

auto EffectSystem::free_buffer_memory(std::string_view name) -> void {
    if (gpu_ != nullptr) {
        gpu_->release_persistent_buffer(name);
    }
}

auto EffectSystem::declare(GameGraph &graph) -> void {
    if (graph.slot() == GameSlot::frame_start) {
        if (frames_++ % manifest_poll_frames == 0) {
            poll_manifests();
        }
        for (auto const &name: unused_buffers_) {
            free_buffer_memory(name);
        }
        unused_buffers_.clear();
    }
    if (order_.empty()) {
        return;
    }
    ZoneScopedNC("Effects declare", tracy::Color::YellowGreen);

    // A copy: dropping an effect from a frame must not disturb the walk.
    for (auto const id: std::vector<EffectId>{order_}) {
        auto const found = effects_.find(id);
        if (found == effects_.end() || found->second.slot != graph.slot()) {
            continue;
        }
        auto &effect = found->second;

        auto const reason = graph.isolated(std::format("effect {}", id), [&] { declare_effect(graph, id, effect); });
        if (reason) {
            effect.problem = *reason;
        } else if (!effect.problem.empty() && effect.problem.starts_with("Game frame graph")) {
            effect.problem.clear();
        }
    }
}

auto EffectSystem::declare_effect(GameGraph &graph, EffectId id, Effect &effect) -> void {
    auto const &shader = shaders_[effect.shader];
    auto const &manifest = shader.manifest;

    auto params = DynamicParams{};
    auto output = std::optional<std::size_t>{};
    auto replaced = false;
    auto dispatch_elements = std::uint32_t{0};
    auto const label = std::format("effect {}", shader.name);
    // The names an output image is created with are read while the pass is declared, so they live until then.
    auto names = std::vector<std::string>{};
    names.reserve(manifest.bindings.size());

    for (auto index = std::size_t{0}; index < manifest.bindings.size(); ++index) {
        auto const &binding = manifest.bindings[index];

        if (std::holds_alternative<EffectImageIn>(binding.what)) {
            auto source = std::optional<ImageRead>{};
            if (effect.sources[index] == EffectSource::scene_colour) {
                source = graph.scene_colour_source();
            } else if (auto const depth = graph.scene_depth()) {
                source = ImageRead{*depth};
            }
            if (!source) {
                effect.problem = std::format("'{}' reads {}, which is not available in {}", binding.name,
                                             source_name(effect.sources[index]), game_slot_name(graph.slot()));
                return;
            }
            params.members.emplace_back(*source);
        } else if (auto const *image = std::get_if<EffectImageOut>(&binding.what)) {
            names.push_back(std::format("effect.{}.{}", id, binding.name));
            auto write = ImageWrite{GameImageDesc{.format = to_vk_format(image->format), .name = names.back()}};
            if (image->replaces_scene_colour) {
                replaced = true;
                output = params.members.size();
            }
            params.members.emplace_back(std::move(write));
        } else if (auto const *buffer = std::get_if<EffectBufferBinding>(&binding.what)) {
            auto elements = std::uint32_t{0};
            auto handle = GameBuffer{};
            if (effect.buffers[index]) {
                elements = buffer_elements(*effect.buffers[index]);
                handle = graph.persistent_buffer(std::format("effect_buffer.{}", *effect.buffers[index]),
                                                 {.size = elements * VkDeviceSize{4}});
            } else {
                elements = buffer->elements;
                handle = graph.persistent_buffer(owned_buffer_name(id, binding.name),
                                                 {.size = elements * VkDeviceSize{4}});
            }
            if (manifest.dispatch.kind == EffectDispatch::Kind::elements && manifest.dispatch.of == binding.name) {
                dispatch_elements = elements;
            }
            switch (buffer->access) {
                case EffectBufferAccess::read:
                    params.members.emplace_back(BufferRead{handle});
                    break;
                case EffectBufferAccess::write:
                    params.members.emplace_back(BufferWrite{handle});
                    break;
                case EffectBufferAccess::read_write:
                    params.members.emplace_back(BufferReadWrite{handle});
                    break;
            }
        }
    }

    for (auto index = std::size_t{0}; index < manifest.params.size(); ++index) {
        auto const &param = manifest.params[index];
        auto const &value = effect.params[index];
        auto const floats = [&] {
            auto result = std::array<float, 4>{};
            for (auto component = std::size_t{0}; component < 4; ++component) {
                result[component] = static_cast<float>(value[component]);
            }
            return result;
        }();

        switch (param.type) {
            case EffectParamType::float1:
                params.members.emplace_back(PushBytes::of(floats[0]));
                break;
            case EffectParamType::float2:
                params.members.emplace_back(PushBytes::of(std::array{floats[0], floats[1]}, 8));
                break;
            case EffectParamType::float3:
                params.members.emplace_back(PushBytes::of(std::array{floats[0], floats[1], floats[2]}, 16));
                break;
            case EffectParamType::float4:
                params.members.emplace_back(PushBytes::of(floats, 16));
                break;
            case EffectParamType::int1:
                params.members.emplace_back(PushBytes::of(static_cast<std::int32_t>(value[0])));
                break;
            case EffectParamType::uint1:
                params.members.emplace_back(PushBytes::of(static_cast<std::uint32_t>(value[0])));
                break;
        }
    }

    auto threads = Threads{.group_x = manifest.group_size[0], .group_y = manifest.group_size[1]};
    switch (manifest.dispatch.kind) {
        case EffectDispatch::Kind::per_pixel: {
            auto const extent = graph.scene_extent();
            threads.x = extent.width;
            threads.y = extent.height;
            break;
        }
        case EffectDispatch::Kind::threads:
            threads.x = manifest.dispatch.threads;
            break;
        case EffectDispatch::Kind::elements:
            threads.x = dispatch_elements;
            break;
    }

    graph.add_compute(std::format("effect.{}", id), params, shader.compute, threads,
                      {.queue = frame_graph::QueueAffinity::compute_preferred,
                       .side_effect = !replaced,
                       .label = label,
                       .color = effect_colour});

    if (output) {
        graph.replace_scene_colour(std::get<ImageWrite>(params.members[*output]).image);
    }
}
