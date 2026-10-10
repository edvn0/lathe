#include "app/lua_animation.hxx"

#include <algorithm>
#include <array>
#include <format>

#include <glm/gtc/matrix_transform.hpp>

#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

struct LuaAnimationSystem::Rig {
    ModelHandle model;
    RigOptions options;
    entt::entity entity = entt::null;

    // Set by prepare().
    std::shared_ptr<ModelAnimationData const> data;
    std::vector<std::unique_ptr<Animation::KeyframeClip>> clips;
    std::unique_ptr<Animation::LocomotionStateTable> table;
    std::unique_ptr<Animation::AnimStateMachine> machine;
    bool failed = false;

    // Rebuilt when the number of actors changes.
    std::unique_ptr<Animation::AnimationBatch> batch;
    std::size_t batched = 0;

    std::vector<Actor> actors;
};

LuaAnimationSystem::LuaAnimationSystem() = default;
LuaAnimationSystem::~LuaAnimationSystem() = default;

auto LuaAnimationSystem::create_rig(Scene &scene, ModelHandle model, RigOptions const &options) -> std::uint32_t {
    for (std::size_t i = 0; i < rigs_.size(); ++i) {
        if (rigs_[i]->model == model) {
            return static_cast<std::uint32_t>(i);
        }
    }

    auto rig = std::make_unique<Rig>();
    rig->model = model;
    rig->options = options;

    auto const entity = Entity{&scene, std::format("lua_actors_{}", rigs_.size())};
    entity.emplace<Components::Transform>(Components::Transform{});
    entity.emplace<Components::InstancedModel>(Components::InstancedModel{.model = model});
    rig->entity = static_cast<entt::entity>(entity);

    rigs_.push_back(std::move(rig));
    return static_cast<std::uint32_t>(rigs_.size() - 1);
}

auto LuaAnimationSystem::rig_count() const noexcept -> std::size_t { return rigs_.size(); }

auto LuaAnimationSystem::prepare(Renderer const &renderer, Rig &rig) -> bool {
    if (rig.machine != nullptr) {
        return true;
    }

    if (rig.failed) {
        return false;
    }

    auto data = renderer.model_animation(rig.model);

    if (!data) {
        return false;
    }

    auto const make = [&](std::string_view name) -> Animation::Clip const * {
        auto const *imported = data->find_clip(name);

        if (imported == nullptr) {
            return nullptr;
        }

        rig.clips.push_back(imported->make_clip());
        return rig.clips.back().get();
    };

    Animation::LocomotionClipSet set;
    set.idle = make("Idle");

    if (set.idle == nullptr && !data->clips.empty()) {
        rig.clips.push_back(data->clips.front().make_clip());
        set.idle = rig.clips.back().get();
    }

    set.walk = make("Walk");
    set.run = make("Run");
    set.jump = make("Jump");
    set.lie_down = make("Death");

    if (set.idle == nullptr) {
        warn("[lua] the model has a skeleton but no animation clips; its actors cannot be drawn");
        rig.failed = true;
        return false;
    }

    set.walk_stride = set.walk != nullptr ? rig.options.walk_speed * set.walk->duration() : 0.0F;
    set.run_stride = set.run != nullptr ? rig.options.run_speed * set.run->duration() : 0.0F;

    rig.data = std::move(data);
    rig.table = std::make_unique<Animation::LocomotionStateTable>(set);
    rig.machine = std::make_unique<Animation::AnimStateMachine>(rig.table->table());

    return true;
}

auto LuaAnimationSystem::ready(Renderer const &renderer, std::uint32_t rig) -> bool {
    return rig < rigs_.size() && prepare(renderer, *rigs_[rig]);
}

auto LuaAnimationSystem::add_actor(std::uint32_t rig) -> std::uint32_t {
    auto &actors = rigs_[rig]->actors;
    actors.emplace_back();
    return static_cast<std::uint32_t>(actors.size() - 1);
}

auto LuaAnimationSystem::actor(std::uint32_t rig, std::uint32_t index) -> Actor * {
    if (rig >= rigs_.size() || index >= rigs_[rig]->actors.size()) {
        return nullptr;
    }

    return &rigs_[rig]->actors[index];
}

auto LuaAnimationSystem::state_name(std::uint32_t rig, std::uint32_t index) const -> std::string_view {
    constexpr std::array<std::string_view, Animation::state_count> names{
            "idle", "walk", "run", "jump_rise", "jump_apex", "jump_fall", "lying_down", "prone", "getting_up",
    };

    if (rig >= rigs_.size() || rigs_[rig]->batch == nullptr || index >= rigs_[rig]->batched) {
        return {};
    }

    return names[static_cast<std::size_t>(rigs_[rig]->batch->state(index).current)];
}

auto LuaAnimationSystem::update(Scene &scene, Renderer &renderer, float delta_time) -> void {
    if (rigs_.empty()) {
        return;
    }

    static_cast<void>(renderer.set_skin_palette({}));

    auto &registry = scene.get_registry();

    for (auto &rig_pointer: rigs_) {
        auto &rig = *rig_pointer;

        if (!registry.valid(rig.entity) || !registry.all_of<Components::InstancedModel>(rig.entity)) {
            continue;
        }

        auto &instanced = registry.get<Components::InstancedModel>(rig.entity);
        instanced.transforms.clear();
        instanced.palette_offsets.clear();

        if (!prepare(renderer, rig) || rig.actors.empty()) {
            instanced.touch();
            continue;
        }

        if (rig.batch == nullptr || rig.batched != rig.actors.size()) {
            rig.batch = std::make_unique<Animation::AnimationBatch>(rig.data->skeleton, *rig.machine,
                                                                    rig.actors.size(), &thread_pool());
            rig.batched = rig.actors.size();
        }

        std::vector<Animation::AnimInputs> inputs;
        inputs.reserve(rig.actors.size());

        for (auto const &actor: rig.actors) {
            inputs.push_back(actor.inputs);
        }

        rig.batch->update(inputs, delta_time);

        auto const &options = rig.options;
        auto const fit = glm::translate(glm::scale(glm::mat4{1.0F}, glm::vec3{options.height / options.model_height}),
                                        {0.0F, -options.model_feet_y, 0.0F});
        auto const facing = glm::rotate(glm::mat4{1.0F}, options.yaw_offset, glm::vec3{0.0F, 1.0F, 0.0F});

        // Instances past the renderer's skin budget would draw unskinned, so none are asked for.
        auto const budget = renderer.max_skinned_instances(rig.model);

        for (std::size_t i = 0; i < rig.actors.size() && instanced.transforms.size() < budget; ++i) {
            auto const &actor = rig.actors[i];

            if (!actor.active) {
                continue;
            }

            auto const offset = renderer.append_skin_palette(rig.batch->palette(i));

            if (!offset) {
                break;
            }

            auto const root = glm::rotate(glm::translate(glm::mat4{1.0F}, actor.position), actor.yaw,
                                          glm::vec3{0.0F, 1.0F, 0.0F});

            instanced.transforms.push_back(root * facing * fit);
            instanced.palette_offsets.push_back(*offset);
        }

        instanced.touch();
    }
}
